"""torchrun worker: one vLLM rank's source map, on CPU.

vLLM's world is ``dp x pp x prefill_cp x tp``. Expert parallel (``--ep``) splits
whole experts across that group instead of tensor-sharding each expert. Ranks run
the same ``load_weights`` and post-processing as ``CpuVllm``. ``--real`` saves the
loaded parameters for verification.

    torchrun --nproc-per-node 2 -m lumenrl.tools.vllm_shard_maps \\
        --model /data/rl_data/models/Qwen3-8B-Base --out /tmp/maps --tp 2
"""

from __future__ import annotations

import argparse
import os
import time

import torch

from lumenrl.tools.build_weight_map import CpuVllm
from lumenrl.transfer.weight_source_map import CheckpointIndex, capture


def _coords(args, rank: int) -> dict:
    """This rank's place in vLLM's ``dp x pp x pcp x tp`` layout."""
    tp_rank = rank % args.tp
    rest = rank // args.tp
    pcp_rank = rest % args.pcp
    rest //= args.pcp
    pp_rank = rest % args.pp
    dp_rank = rest // args.pp
    return {"rank": rank, "world": int(os.environ["WORLD_SIZE"]),
            "tp_rank": tp_rank, "pp_rank": pp_rank, "pcp_rank": pcp_rank,
            "dp_rank": dp_rank, "ep": bool(args.ep)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--dp", type=int, default=1)
    ap.add_argument("--pcp", type=int, default=1, help="prefill context parallel")
    ap.add_argument("--dcp", type=int, default=1, help="decode context parallel, divides TP")
    ap.add_argument("--ep", action="store_true", help="enable_expert_parallel")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    rank = int(os.environ["RANK"])
    world = args.tp * args.pp * args.dp * args.pcp
    if world != int(os.environ["WORLD_SIZE"]):
        raise SystemExit(f"world {os.environ['WORLD_SIZE']} != tp*pp*dp*pcp = {world}")

    vm = CpuVllm(args.model, tp=args.tp, pp=args.pp, dp=args.dp, pcp=args.pcp, dcp=args.dcp,
                 expert_parallel=args.ep, rank=rank, world=world)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    coords = _coords(args, rank)
    if args.real:
        _, final = vm.load(vm.real_weights(), keep_loader_stage=False)
        torch.save({"coords": coords, "params": {n: p.detach().clone() for n, p in final.items()}},
                   os.path.join(args.out, f"real_rank{rank}.pt"))
    else:
        index = CheckpointIndex.from_dir(args.model)

        def run(k):
            loader, final = vm.load(index.id_tensors(k), keep_loader_stage=True)
            return {"loader": loader, "final": final}

        log = (lambda s: print(f"[vllm {rank}] {s}", flush=True)) if rank == 0 else (lambda s: None)
        maps, classes = capture(index, run, compare=("loader", "final"), log=log)
        m = maps["loader"]
        m.meta["coords"] = coords
        m.meta["classes"] = classes
        m.save(os.path.join(args.out, f"vllm_rank{rank}.pt"))
        print(f"[vllm {rank}] {coords} {m.stats()} {time.time() - t0:.1f}s", flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
