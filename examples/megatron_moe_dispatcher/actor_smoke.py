"""8-GPU Megatron-Native Qwen3-30B-A3B actor smoke: EP=8 A2A or MORI + step profiler."""
from __future__ import annotations

import os
import time

_local = os.environ["LOCAL_RANK"]
# MORI-SHMEM needs hipDeviceCanAccessPeer to every intra-node GPU. Pinning
# CUDA_VISIBLE_DEVICES to a single ordinal makes those peer IDs invalid.
# MORI always needs all 8 visible. For an apples-to-apples A2A-vs-MORI compare
# (matching Megatron's train_qwen3.sh, which exports HIP_VISIBLE_DEVICES=0..7 for
# every dispatcher), KEEP_ALL_GPUS=1 keeps A2A on the same 8-visible NCCL topology
# instead of the per-rank-pinned fast path.
_keep_all_gpus = (
    os.environ.get("MOE_DISPATCHER", "alltoall") == "flex"
    or os.environ.get("KEEP_ALL_GPUS", "0") in ("1", "true", "TRUE")
)
if _keep_all_gpus:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _local
    os.environ["LOCAL_RANK"] = "0"
os.environ.pop("HIP_VISIBLE_DEVICES", None)
os.environ.pop("ROCR_VISIBLE_DEVICES", None)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from megatron.core.jit import disable_jit_fuser

# Flex dispatcher's @jit_fuser uses torch.compile/inductor; some ROCm Triton
# builds lack triton_key. Disable before importing Megatron MoE modules.
disable_jit_fuser()

import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile

from lumenrl.core.protocol import DataProto
from lumenrl.workers.actor_worker import LumenActorWorker

PROFILE_DIR = os.environ.get("PROFILE_DIR", "/workspace/data/logs/actor_profile")
MODEL = os.environ.get("MODEL_PATH", "/workspace/data/models/Qwen3-30B-A3B-Base")
SEQ_LEN = int(os.environ.get("SEQ_LEN", "8192"))
N_ROWS = int(os.environ.get("N_ROWS", "1"))
PROMPT_LEN = int(os.environ.get("PROMPT_LEN", "2048"))
N_STEPS = int(os.environ.get("N_STEPS", "20"))
PROFILE_STEP = int(os.environ.get("PROFILE_STEP", "12"))
MAX_TOKENS_PER_GPU = int(os.environ.get("MAX_TOKENS_PER_GPU", "8192"))
DISPATCHER = os.environ.get("MOE_DISPATCHER", "alltoall")
FLEX_BACKEND = os.environ.get("MOE_FLEX_BACKEND", "mori")
TRACE_TAG = os.environ.get("TRACE_TAG", DISPATCHER)
# Example 7 YAML sets moe_grouped_gemm=true; ROCm SequentialMLP is the
# working path on gfx942 (CUTLASS grouped GEMM is not).
MOE_GROUPED_GEMM = os.environ.get("MOE_GROUPED_GEMM", "0") in ("1", "true", "TRUE")


def _batch(world: int) -> DataProto:
    b, s = N_ROWS, SEQ_LEN
    prompt = min(PROMPT_LEN, s)
    ids = torch.randint(10, 1000, (b, s), dtype=torch.long)
    am = torch.ones(b, s, dtype=torch.long)
    resp = torch.ones(b, s, dtype=torch.long)
    resp[:, :prompt] = 0
    return DataProto(
        tensors={"input_ids": ids, "attention_mask": am, "response_mask": resp},
        meta={"calculate_entropy": False, "temperature": 1.0},
    )


def _train_batch(ids, am, resp, old_lp, world: int) -> DataProto:
    tok = int((resp[:, 1:] == 1).sum().item())
    return DataProto(
        tensors={
            "input_ids": ids,
            "attention_mask": am,
            "response_mask": resp,
            "old_log_probs": old_lp,
            "advantages": torch.ones_like(old_lp),
        },
        meta={
            "algorithm": "dapo",
            "temperature": 1.0,
            "batch_num_tokens": float(tok * world),
            "dp_size": world,
            "algo_config": {
                "dapo": {
                    "clip_ratio_low": 0.2,
                    "clip_ratio_high": 0.28,
                    "kl_coeff": 0.0,
                }
            },
        },
    )


def main() -> None:
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    rank = dist.get_rank()
    world = dist.get_world_size()
    os.makedirs(PROFILE_DIR, exist_ok=True)

    worker = LumenActorWorker(
        rank,
        world,
        {
            "policy": {
                "model_name": MODEL,
                "training_backend": "megatron_native",
                "seed": 42,
                "learning_rate": 1.0e-6,
                "max_grad_norm": 1.0,
                "weight_decay": 0.1,
                "lr_warmup_steps": 0,
                "training": {
                    "optimizer_dtype": "bf16",
                    "megatron_cfg": {
                        "use_distributed_optimizer": True,
                        "ddp_bucket_size": int(os.environ.get("DDP_BUCKET_SIZE", "0")) or None,
                        "overlap_grad_reduce": os.environ.get("OVERLAP_GRAD_REDUCE", "0") in ("1", "true", "TRUE"),
                        "overlap_param_gather": os.environ.get("OVERLAP_PARAM_GATHER", "0") in ("1", "true", "TRUE"),
                        "tensor_model_parallel_size": 1,
                        "pipeline_model_parallel_size": 1,
                        "context_parallel_size": 1,
                        "expert_model_parallel_size": 8,
                        "sequence_parallel": False,
                        "moe_grouped_gemm": MOE_GROUPED_GEMM,
                        "moe_permute_fusion": True,
                        "moe_token_dispatcher_type": DISPATCHER,
                        "moe_flex_dispatcher_backend": FLEX_BACKEND,
                        "moe_aux_loss_coeff": float(os.environ.get("MOE_AUX_LOSS", "0.0")),
                        "moe_router_dtype": "fp32",
                        "recompute_granularity": "full",
                        "recompute_method": "uniform",
                        "recompute_num_layers": 1,
                        "log_probs_chunk_size": 1024,
                        "enable_dynamic_batch": True,
                        "max_tokens_per_gpu": MAX_TOKENS_PER_GPU,
                    },
                },
            },
        },
    )
    t0 = time.perf_counter()
    worker.init_model()
    dist.barrier()
    if rank == 0:
        print(
            f"init_model {time.perf_counter() - t0:.1f}s world={world} "
            f"dispatcher={DISPATCHER} flex={FLEX_BACKEND} ep=8 "
            f"seq={SEQ_LEN} rows={N_ROWS} prompt={PROMPT_LEN} "
            f"max_tokens_per_gpu={MAX_TOKENS_PER_GPU}",
            flush=True,
        )
        print(
            f"MEM allocated={torch.cuda.memory_allocated()/2**30:.1f}GiB "
            f"reserved={torch.cuda.memory_reserved()/2**30:.1f}GiB",
            flush=True,
        )

    proto = _batch(world)
    t1 = time.perf_counter()
    out = worker.compute_log_probs(proto)
    dist.barrier()
    if rank == 0:
        print(f"compute_log_probs {time.perf_counter() - t1:.2f}s", flush=True)

    old = out.tensors["log_probs"]
    train = _train_batch(
        proto.tensors["input_ids"],
        proto.tensors["attention_mask"],
        proto.tensors["response_mask"],
        old,
        world,
    )

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    prof = None
    for step in range(N_STEPS):
        capture = step == PROFILE_STEP
        if capture:
            torch.cuda.synchronize()
            dist.barrier()
            prof = profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
            prof.__enter__()
        t = time.perf_counter()
        metrics = worker.update_policy(train)
        dist.barrier()
        if capture:
            torch.cuda.synchronize()
            prof.__exit__(None, None, None)
        dt = time.perf_counter() - t
        if rank == 0:
            tag = " profiled" if capture else ""
            print(f"step {step} {dt:.2f}s{tag} {metrics}", flush=True)

    if prof is not None and rank in (0, 1):
        trace = os.path.join(
            PROFILE_DIR, f"{TRACE_TAG}_step{PROFILE_STEP}_actor_rank{rank}.json"
        )
        prof.export_chrome_trace(trace)
        table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30)
        summary = os.path.join(
            PROFILE_DIR, f"{TRACE_TAG}_step{PROFILE_STEP}_actor_rank{rank}.txt"
        )
        with open(summary, "w") as fh:
            fh.write(table)
        if rank == 0:
            print(table, flush=True)
            print(f"chrome trace (step {PROFILE_STEP}): {trace}", flush=True)
            print(f"PASS: 8-GPU Megatron-Native MoE {TRACE_TAG} actor smoke", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
