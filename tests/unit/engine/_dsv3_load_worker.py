"""torchrun worker for the DSv3 load test. Not collected by pytest.

Loads a random full HF state into the tiny DSv3 model at ``DSV3_TP``/``DSV3_EP``
through the engine's own sharded loader (``_shard_hf_for_moe``), then gathers it
back through the engine's export (``_export_dsv3``) and requires every HF tensor,
the router bias included, to come back bit-identical. ``DSV3_Q_LORA=0`` uses the
full-rank q projection (Moonlight, DeepSeek-V2-Lite).

Prints ``DSV3_LOAD_OK`` on success.
"""

import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.environ["LUMENRL_ROOT"])

from megatron.core import parallel_state as mpu  # noqa: E402
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec  # noqa: E402
from megatron.core.models.gpt.gpt_model import GPTModel  # noqa: E402
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # noqa: E402

from lumenrl.engine.training.bridges import dsv3  # noqa: E402
from lumenrl.engine.training.megatron_native_engine import MegatronNativeEngine  # noqa: E402
from lumenrl.engine.training.model_specs import DSV3, _export_dsv3  # noqa: E402

TP = int(os.environ.get("DSV3_TP", 2))
EP = int(os.environ.get("DSV3_EP", 2))
Q_LORA = int(os.environ.get("DSV3_Q_LORA", 32)) or None

H, HEADS, NOPE, ROPE, V, KV_LORA = 128, 4, 16, 8, 16, 16
FFN, MOE_FFN, E, SHARED, VOCAB, LAYERS = 256, 64, 4, 1, 512, 3
TINY = {
    "architectures": ["DeepseekV3ForCausalLM"],
    "num_hidden_layers": LAYERS, "hidden_size": H, "num_attention_heads": HEADS,
    "num_key_value_heads": HEADS, "intermediate_size": FFN, "vocab_size": VOCAB,
    "rms_norm_eps": 1e-6, "q_lora_rank": Q_LORA, "kv_lora_rank": KV_LORA,
    "qk_nope_head_dim": NOPE, "qk_rope_head_dim": ROPE, "v_head_dim": V,
    "n_routed_experts": E, "n_shared_experts": SHARED, "num_experts_per_tok": 2,
    "moe_intermediate_size": MOE_FFN, "first_k_dense_replace": 1,
    "routed_scaling_factor": 2.5, "topk_method": "noaux_tc", "scoring_func": "sigmoid",
}


def hf_shapes() -> dict:
    s = {"model.embed_tokens.weight": (VOCAB, H), "model.norm.weight": (H,),
         "lm_head.weight": (VOCAB, H)}
    for L in range(LAYERS):
        p = f"model.layers.{L}."
        a = p + "self_attn."
        if Q_LORA:
            s |= {a + "q_a_proj.weight": (Q_LORA, H), a + "q_a_layernorm.weight": (Q_LORA,),
                  a + "q_b_proj.weight": (HEADS * (NOPE + ROPE), Q_LORA)}
        else:
            s[a + "q_proj.weight"] = (HEADS * (NOPE + ROPE), H)
        s |= {a + "kv_a_proj_with_mqa.weight": (KV_LORA + ROPE, H),
              a + "kv_a_layernorm.weight": (KV_LORA,),
              a + "kv_b_proj.weight": (HEADS * (NOPE + V), KV_LORA),
              a + "o_proj.weight": (H, HEADS * V),
              p + "input_layernorm.weight": (H,), p + "post_attention_layernorm.weight": (H,)}
        if L == 0:
            s |= {p + "mlp.gate_proj.weight": (FFN, H), p + "mlp.up_proj.weight": (FFN, H),
                  p + "mlp.down_proj.weight": (H, FFN)}
            continue
        s |= {p + "mlp.gate.weight": (E, H), p + "mlp.gate.e_score_correction_bias": (E,)}
        sh = p + "mlp.shared_experts."
        s |= {sh + "gate_proj.weight": (MOE_FFN * SHARED, H),
              sh + "up_proj.weight": (MOE_FFN * SHARED, H),
              sh + "down_proj.weight": (H, MOE_FFN * SHARED)}
        for e in range(E):
            x = f"{p}mlp.experts.{e}."
            s |= {x + "gate_proj.weight": (MOE_FFN, H), x + "up_proj.weight": (MOE_FFN, H),
                  x + "down_proj.weight": (H, MOE_FFN)}
    return s


def main() -> None:
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    mpu.initialize_model_parallel(TP, 1, expert_model_parallel_size=EP,
                                  expert_tensor_parallel_size=1)
    model_parallel_cuda_manual_seed(0)

    cfg = dsv3.build_dsv3_config(TINY, {"moe_grouped_gemm": True}, tp=TP, ep=EP, etp=1)
    spec = get_gpt_decoder_block_spec(cfg, use_transformer_engine=True)
    model = GPTModel(config=cfg, transformer_layer_spec=spec, vocab_size=VOCAB,
                     max_sequence_length=64, share_embeddings_and_output_weights=False,
                     position_embedding_type="rope", parallel_output=False).cuda()

    g = torch.Generator().manual_seed(1234)
    hf = {k: torch.randn(shape, generator=g).to(torch.bfloat16) for k, shape in hf_shapes().items()}

    eng = object.__new__(MegatronNativeEngine)
    eng._dims, eng._spec, eng.module = dsv3.build_dsv3_dims(TINY), DSV3, model
    eng._num_experts, eng._is_moe = E, True
    eng._tp, eng._pp, eng._cp, eng._ep, eng._etp = TP, 1, 1, EP, 1
    missing = model.load_state_dict(eng._shard_hf_for_moe(model, hf), strict=False)
    real = [k for k in missing.missing_keys if "_extra_state" not in k]
    assert not real, f"missing keys after load: {real[:6]}"

    back = {k: v.detach().cpu() for k, v in _export_dsv3(eng)}
    lost = sorted(set(hf) - set(back))
    assert not lost, f"export lost {len(lost)}: {lost[:6]}"
    for k, want in hf.items():
        got = back[k].to(want.dtype)
        assert got.shape == want.shape, f"{k}: {tuple(got.shape)} != {tuple(want.shape)}"
        assert torch.equal(got, want), f"{k}: values differ after load + export"
    dist.barrier()
    if dist.get_rank() == 0:
        print(f"DSV3_LOAD_OK tp={TP} ep={EP} q_lora={Q_LORA} tensors={len(hf)}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
