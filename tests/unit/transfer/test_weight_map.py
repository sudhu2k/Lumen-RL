"""Id-checkpoint source maps and transfer plans on toy loaders (CPU only)."""

import pytest
import torch

from lumenrl.engine.training.bridges.gpt import GPTDims, hf_to_megatron
from lumenrl.transfer.weight_plan import apply_plan, correspond, plan_rounds
from lumenrl.transfer.weight_source_map import (
    CheckpointIndex,
    CheckpointTensor,
    ParamMap,
    capture,
    checkpoint_map,
)

D = GPTDims(num_layers=2, hidden=8, num_heads=4, num_kv_groups=2, head_dim=2, ffn=6, vocab=40)
VOCAB_PAD = 48


def _index(base: int) -> CheckpointIndex:
    shapes = {"model.embed_tokens.weight": (D.vocab, D.hidden), "model.norm.weight": (D.hidden,),
              "lm_head.weight": (D.vocab, D.hidden)}
    for i in range(D.num_layers):
        p = f"model.layers.{i}."
        shapes.update({
            p + "input_layernorm.weight": (D.hidden,),
            p + "post_attention_layernorm.weight": (D.hidden,),
            p + "self_attn.q_proj.weight": (D.num_heads * D.head_dim, D.hidden),
            p + "self_attn.k_proj.weight": (D.num_kv_groups * D.head_dim, D.hidden),
            p + "self_attn.v_proj.weight": (D.num_kv_groups * D.head_dim, D.hidden),
            p + "self_attn.o_proj.weight": (D.hidden, D.num_heads * D.head_dim),
            p + "mlp.gate_proj.weight": (D.ffn, D.hidden),
            p + "mlp.up_proj.weight": (D.ffn, D.hidden),
            p + "mlp.down_proj.weight": (D.hidden, D.ffn),
        })
    tensors, start = [], base
    for name, shape in shapes.items():
        n = 1
        for s in shape:
            n *= s
        tensors.append(CheckpointTensor(name, shape, "BF16", start, n))
        start += n
    return CheckpointIndex(tensors)


def _trainer(hf):
    return hf_to_megatron(hf, D, te=True)


def _rollout(hf, tp_rank=0, tp=1):
    """vLLM-like layout: stacked [q;k;v], [gate;up], padded vocab, column-parallel o/down."""
    out = {}
    emb = torch.zeros((VOCAB_PAD, D.hidden), dtype=hf["model.embed_tokens.weight"].dtype)
    emb[:D.vocab] = hf["model.embed_tokens.weight"]
    out["model.embed_tokens.weight"] = emb
    out["model.norm.weight"] = hf["model.norm.weight"].clone()
    out["lm_head.weight"] = hf["lm_head.weight"].clone()
    for i in range(D.num_layers):
        p = f"model.layers.{i}."
        out[p + "self_attn.qkv_proj.weight"] = torch.cat(
            [hf[p + f"self_attn.{x}_proj.weight"] for x in "qkv"]).contiguous()
        o = hf[p + "self_attn.o_proj.weight"]
        w = o.shape[1] // tp
        out[p + "self_attn.o_proj.weight"] = o[:, tp_rank * w:(tp_rank + 1) * w].contiguous()
        out[p + "mlp.gate_up_proj.weight"] = torch.cat(
            [hf[p + "mlp.gate_proj.weight"], hf[p + "mlp.up_proj.weight"]]).contiguous()
        out[p + "mlp.down_proj.weight"] = hf[p + "mlp.down_proj.weight"].clone()
        out[p + "input_layernorm.weight"] = hf[p + "input_layernorm.weight"].clone()
        out[p + "post_attention_layernorm.weight"] = hf[p + "post_attention_layernorm.weight"].clone()
    return out


def _tile(t: torch.Tensor) -> torch.Tensor:
    """A post-processing permutation: 2-row x 4-column tiles, tile-major."""
    r, c = t.shape
    return t.view(r // 2, 2, c // 4, 4).permute(0, 2, 1, 3).contiguous().view(r, c)


def _truth(index, loader):
    hf = {t.name: torch.arange(t.start, t.start + t.numel).view(t.shape) for t in index.tensors}
    return loader(hf)


@pytest.mark.parametrize("start", [0, 1, 255, 256, 65_530, (1 << 24) - 3, (1 << 33) + 77,
                                   (1 << 40) - 5])
@pytest.mark.parametrize("n", [0, 1, 7, 256, 300, 70_000])
def test_id_tensor_is_byte_k_of_ids(start, n):
    index = CheckpointIndex([CheckpointTensor("t", (n,), "BF16", start, n)])
    ids = torch.arange(start, start + n, dtype=torch.int64)
    for k in range(6):
        assert torch.equal(index.id_tensor("t", k), ((ids >> (8 * k)) & 255).to(torch.bfloat16))


@pytest.mark.parametrize("base", [0, (1 << 33) + 77])
def test_decoded_ids_match_loader_on_int64_ids(base):
    index = _index(base)
    maps, _ = capture(index, lambda k: {"loader": _trainer(index.id_state(k))}, log=lambda s: None)
    truth = _truth(index, _trainer)
    m = maps["loader"]
    assert set(m.params) == set(truth)
    for name, t in truth.items():
        assert torch.equal(m.params[name].ids(), t.reshape(-1)), name
    qkv = m.params["decoder.layers.0.self_attention.linear_qkv.weight"]
    assert len(qkv.runs.as_lines()) == D.num_kv_groups * 3


def test_padding_and_transpose():
    index = _index(1 << 20)

    def load(hf):
        out = _rollout(hf)
        out["t"] = hf["model.layers.0.mlp.down_proj.weight"].t().contiguous()
        return out

    maps, _ = capture(index, lambda k: {"loader": load(index.id_state(k))}, log=lambda s: None)
    truth = _truth(index, load)
    emb = maps["loader"].params["model.embed_tokens.weight"].ids()
    assert torch.equal(emb[:D.vocab * D.hidden], truth["model.embed_tokens.weight"][:D.vocab].reshape(-1))
    assert bool((emb[D.vocab * D.hidden:] == -1).all())
    t = maps["loader"].params["t"]
    assert torch.equal(t.ids(), truth["t"].reshape(-1))
    assert set(t.runs.as_lines().ckpt_strides[:, 0].tolist()) == {D.ffn}


def test_classification_identity_permutation_changed_computed():
    index = _index(5000)
    down = "model.layers.0.mlp.down_proj.weight"
    shuffle = torch.randperm(D.hidden * D.ffn, generator=torch.Generator().manual_seed(0))

    def run(k):
        loader = _rollout(index.id_state(k))
        final = {n: t.clone() for n, t in loader.items()}
        final["model.layers.0.mlp.gate_up_proj.weight"] = _tile(final["model.layers.0.mlp.gate_up_proj.weight"])
        final[down] = final[down].reshape(-1)[shuffle].view(final[down].shape)
        final["model.norm.weight"] = final["model.norm.weight"] * 0.5
        return {"loader": loader, "final": final}

    _, classes = capture(index, run, compare=("loader", "final"), log=lambda s: None)
    assert classes["model.layers.0.mlp.gate_up_proj.weight"] == "permutation"
    assert classes[down] == "changed"
    assert classes["model.norm.weight"] == "computed"
    assert classes["model.layers.1.mlp.gate_up_proj.weight"] == "identity"


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_plan_reproduces_rollout_layout_from_trainer_layout(tp_rank):
    index = _index((1 << 33) + 5)
    rollout = lambda hf: _rollout(hf, tp_rank=tp_rank, tp=2)  # noqa: E731
    tmap, _ = capture(index, lambda k: {"loader": _trainer(index.id_state(k))}, log=lambda s: None)
    vmap, _ = capture(index, lambda k: {"loader": rollout(index.id_state(k))}, log=lambda s: None)
    copies = correspond(tmap["loader"], vmap["loader"])
    rep = copies.report
    assert not rep["missing_elements"] and not rep["unsupported_dst"] and not rep["dtype_mismatch"]
    plan = plan_rounds(copies, n_src=4, n_dst=3, round_bytes=1024, max_piece_bytes=256)

    torch.manual_seed(0)
    hf = {t.name: torch.randn(t.shape).to(torch.bfloat16) for t in index.tensors}
    trainer = _trainer(hf)
    ref = rollout(hf)
    tr = plan.traffic()
    assert sum(tr["per_src"]) == 3 * plan.meta["model_bytes"]
    assert max(tr["per_src"]) - min(tr["per_src"]) <= 2 * 256
    for d in range(3):
        got = {n: torch.zeros_like(t) for n, t in ref.items()}
        written = apply_plan(plan, d, {s: trainer for s in range(4)}, got)
        for n, t in ref.items():
            if n == "model.embed_tokens.weight":
                assert torch.equal(got[n][:D.vocab], t[:D.vocab])
                assert written[n] == D.vocab * D.hidden * 2
            else:
                assert torch.equal(got[n], t), n
                assert written[n] == t.numel() * 2, n


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_checkpoint_map_is_what_the_loader_reads(tp_rank):
    """Online quantization: the plan reads into checkpoint tensors, exactly the elements
    the rollout rank's loader takes (half of o_proj here; not lm_head, which it drops;
    a raw-id parameter too)."""
    index = _index((1 << 33) + 5)
    down = "model.layers.1.mlp.down_proj.weight"
    shuffle = torch.randperm(D.hidden * D.ffn, generator=torch.Generator().manual_seed(1))

    def rollout(hf):
        out = _rollout(hf, tp_rank=tp_rank, tp=2)
        del out["lm_head.weight"]
        return out

    tmap, _ = capture(index, lambda k: {"loader": _trainer(index.id_state(k))}, log=lambda s: None)
    vmap, _ = capture(index, lambda k: {"loader": rollout(index.id_state(k))}, log=lambda s: None)
    m = vmap["loader"]
    p = m.params[down]
    m.params[down] = ParamMap(down, p.shape, p.dtype, raw=p.ids().view(-1)[shuffle])
    ckpt = checkpoint_map(m, index)

    truth = _truth(index, rollout)
    read = torch.unique(torch.cat([t.reshape(-1) for t in truth.values()]))
    assert set(ckpt.params) == {t.name for t in index.tensors} - {"lm_head.weight"}
    for t in index.tensors:
        if t.name not in ckpt.params:
            continue
        ids = torch.arange(t.start, t.start + t.numel)
        want = torch.where(torch.isin(ids, read), ids, torch.full_like(ids, -1))
        assert torch.equal(ckpt.params[t.name].runs.expand(t.numel), want), t.name
    o = ckpt.params["model.layers.0.self_attn.o_proj.weight"].runs
    assert len(o) == D.hidden and bool((o.length == D.num_heads * D.head_dim // 2).all())

    copies = correspond(tmap["loader"], ckpt)
    assert not copies.report["missing_elements"]
    plan = plan_rounds(copies, n_src=2, n_dst=1, round_bytes=1024, max_piece_bytes=256)
    torch.manual_seed(0)
    hf = {t.name: torch.randn(t.shape).to(torch.bfloat16) for t in index.tensors}
    got = {n: torch.zeros_like(hf[n]) for n in ckpt.params}
    written = apply_plan(plan, 0, {s: _trainer(hf) for s in range(2)}, got)
    for t in index.tensors:
        if t.name in got:
            mask = (ckpt.params[t.name].runs.expand(t.numel) >= 0).view(t.shape)
            assert torch.equal(got[t.name], torch.where(mask, hf[t.name], 0)), t.name
            assert written[t.name] == int(mask.sum()) * 2
