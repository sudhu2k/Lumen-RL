# Monaco MORI / siphon weight-sync results

**Cluster:** Monaco 2-node (R=`10.235.225.36` rollout / Ray head, T=`10.235.225.89` trainer). No shared NFS; plan cache mirrored over SSH.  
**Model:** Qwen3-30B-A3B-Base, DAPO smoke, 4 steps, BF16.  
**Metric:** warm average of `timing/weight_sync_siphon_transfer_s` (drop first sample if it is > 1.5× the median). Startup check is separate and not included.  
**Software (successful runs):**

| Piece | Rev / note |
|---|---|
| Image | `lumenrl:rocm10.3-gfx942-950-20261119` (ROCm 10.3 / HIP 7.18) |
| Siphon | `rocm-siphon` `d531704` + dirty (FSDP side: `adapters/trainers/megatron/fsdp.py`, `runtime.gather_shards`) |
| Lumen-RL DDP | `sudhu/siphon-RL-weight-sync` / synced as `Lumen-RL-siphon` |
| Lumen-RL FSDP8 | `sudhu/siphon-on-megatron-fsdp2` @ `7dbc436` + dirty (`MegatronFsdpSide` in `siphon_call`, `LUMENRL_KEEP_RAY_CLUSTER` in `run_dapo.sh`) |
| MORI knobs | `LUMEN_MORI_NODE_PACK=0` (strided / no mirror packing), `LUMEN_MORI_NIC_MATCH=node` |

Monaco usage for this sweep is done. FSDP8 layouts `v2e` / `v2` / `v1` were **not** finished on Monaco.

---

## 1. Summary tables

### 1.1 DDP trainer × vLLM (siphon)

Trainer parallelisms below are classic Megatron DDP (not FSDP). Crusoe column is the earlier MORI/siphon reference on Crusoe 2-node (same 30B model).

| Trainer | vLLM | Crusoe (s) | Monaco (s) | Monaco / Crusoe | Monaco job (preferred) |
|---|---|---:|---:|---:|---|
| TP4 EP8 | TP8 EP8 | 0.180 | **0.161** | 0.89× | `run-monaco-table-20261122-001317` |
| TP4 EP8 | TP8 | 0.187 | **0.172** | 0.92× | `run-monaco-smoke-20261120-234312` |
| TP4 EP8 | TP4 EP4 × 2 | 0.194 | **0.298** | 1.54× | `run-monaco-smoke-20261121-001552` |
| TP4 EP8 | TP4 × 2 | 0.182 | **0.254** | 1.40× | `run-monaco-smoke-20261121-011832` |
| TP4 EP8 | TP2 EP2 × 4 | 0.296 | **0.311** | 1.05× | `run-monaco-table-20261121-215335` |
| TP4 EP8 | TP2 × 4 | (open / not settled on Crusoe) | **0.515** | — | several table jobs; see §2 |
| TP4 EP8 | TP1 × 8 | 0.567 | **0.390** | 0.69× | several table jobs; see §2 |
| TP2 EP8 | TP8 EP8 | 0.178 | **0.161** | 0.90× | `run-monaco-table-20261122-063613` |
| TP1 EP8 | TP8 EP8 | 0.182 | **0.167** | 0.92× | `run-monaco-table-20261122-084531` |
| TP4 EP4 | TP8 EP8 | 0.187 | **0.201** | 1.07× | `run-monaco-table-20261122-091402` |
| TP2 EP4 | TP8 EP8 | 0.187 | **0.197** | 1.05× | `run-monaco-table-20261122-094617` |
| TP2 PP2 EP4 | TP8 EP8 | 0.182 | **0.197** | 1.08× | `run-monaco-table-20261122-101809` |

**Takeaways (DDP):**

- On **8-GPU engines** (TP8 / TP8 EP8) Monaco is **faster** than Crusoe.
- On **4-GPU engines** (TP4 / TP4 EP4 × 2) Monaco is **slower** (~1.4–1.5×).
- At **TP1 × 8** Monaco is much faster than the Crusoe 0.567 s figure.

### 1.2 FSDP8 trainer × vLLM (siphon)

Trainer = **pure FSDP8**: TP1 EP1 PP1, `data_parallel=megatron_fsdp`, `optim_grads_params` (DP=8). No trainer TP/EP mixing.

Crusoe “MORI, full sharding” numbers below are from the **older in-tree MORI path** (`sudhu/mori-io-weight-transfer` / design.md §7.3) with trainer **TP4 EP8 + Megatron FSDP**, **not** FSDP8. They are **not** a like-for-like trainer layout.

| vLLM | Monaco FSDP8 (s) | Samples | Job | Crusoe full-sharding (s)† | Monaco FSDP8 / Crusoe† |
|---|---:|---:|---|---:|---:|
| TP8 EP8 | **0.222** | 4 | `monaco-fsdp8-20261123-035911` | 0.166 | 1.34× |
| TP8 | **0.187** | 4 | `monaco-fsdp8-20261123-071735` | 0.171–0.172 | ~1.09× |
| TP4 EP4 × 2 | **0.254** | 4 | `monaco-fsdp8-20261123-053112` | 0.176–0.180 | ~1.43× |
| TP4 × 2 | **0.290** | 4 | `monaco-fsdp8-20261123-061721` | 0.173–0.174 | ~1.67× |
| TP2 EP2 × 4 | — | — | **not run** | 0.284–0.296 | — |
| TP2 × 4 | — | — | **not run** | 0.188–0.191 | — |
| TP1 × 8 | — | — | **not run** | 0.334–0.337‡ | — |

† Trainer layout differs (Crusoe: TP4 EP8 + FSDP; Monaco: FSDP8 / TP1 EP1).  
‡ Design.md TP1 row; some pasted Crusoe tables show 0.498–0.580 depending on stage-2 holdback / ZeRO mode.

**Vs Monaco’s own DDP TP4 EP8 column (same cluster):**

| vLLM | Monaco DDP TP4 EP8 | Monaco FSDP8 | FSDP8 / DDP |
|---|---:|---:|---:|
| TP8 EP8 | 0.161 | 0.222 | 1.38× |
| TP8 | 0.172 | 0.187 | 1.09× |
| TP4 EP4 × 2 | 0.298 | 0.254 | **0.85×** (FSDP8 faster) |
| TP4 × 2 | 0.254 | 0.290 | 1.14× |

**Vs other backends (design.md / pasted Crusoe baselines):** Monaco FSDP8 TP8 EP8 (~0.222 s) is still ~15× RCCL (~3.2–3.6 s) and about level with Aqueduct+M2N (~0.224 s). At TP4 EP4 × 2, Monaco FSDP8 (0.254 s) beats Aqueduct+M2N (~0.426 s).

---

## 2. Per-layout detail (warm samples)

### 2.1 DDP — exact preferred-job samples

| Layout | Warm transfers (s) | Mean | Startup check | Exit |
|---|---|---:|---:|---|
| t4e8-v8e | 0.160834, 0.160688, 0.159698, 0.160982 | 0.160550 | yes | 0 |
| t4e8-v8 | 0.171596, 0.173553, 0.170423, 0.174223 | 0.172449 | yes | 0 |
| t4e8-v4e | 0.299455, 0.299539, 0.299178, 0.295066 | 0.298310 | yes | 0 |
| t4e8-v4 | 0.254933, 0.251414, 0.252750, 0.256602 | 0.253925 | yes | 0 |
| t4e8-v2e | 0.313032, 0.310573, 0.307540, 0.308517 | 0.309916 | yes | 0 |
| t4e8-v2 | 0.512757, 0.504923, 0.515126, 0.521098 (`…-044552`) | 0.513476 | yes | 0 |
| t4e8-v2 | 0.517948, 0.510760, 0.513598, 0.515945 (`…-061351`) | 0.514563 | yes | 0 |
| t4e8-v1 | 0.412547, **0.375701**, 0.374076, 0.383664 (`…-041408`, first dropped) | 0.377814 | yes | 0 |
| t4e8-v1 | 0.377027, 0.365149, 0.368870, 0.382389 (`…-054329`) | 0.373359 | yes | 0 |
| t4e8-v1 | 0.364396, 0.357121, 0.397320, 0.383632 (`…-062613`) | 0.375617 | yes | 0 |
| t2e8-v8e | 0.161266, 0.161059, 0.160443, 0.160786 | 0.160888 | yes | 0 |
| t1e8-v8e | 0.166967, 0.166398, 0.166905, 0.166792 | 0.166766 | yes | 0 |
| t4e4-v8e | 0.203055, 0.200530, 0.201488, 0.199125 | 0.201049 | yes | 0 |
| t2e4-v8e | 0.198148, 0.196109, 0.197381, 0.195922 | 0.196890 | yes | 0 |
| t2p2e4-v8e | 0.199323, 0.195615, 0.198493, 0.195263 | 0.197174 | yes (host exit line missing; transfers complete) | — |

Table §1.1 uses rounded means consistent with the filled Monaco DDP column (`t4e8-v2≈0.515`, `t4e8-v1≈0.390` from the multi-run suite report; other rows match the preferred-job means above).

### 2.2 FSDP8 — completed jobs

| Layout | Warm transfers (s) | Mean | Startup check | Exit | Notes |
|---|---|---:|---:|---|---|
| t1e1-v8e | 0.221968, 0.219805, 0.220731, 0.226579 | **0.222271** | 0.630 s | 0 | First successful FSDP8 after siphon FSDP wiring |
| t1e1-v8 | 0.187431, 0.183573, 0.185607, 0.189635 | **0.186561** | yes | 0 | Rerun after incomplete `…-044439` (1 sample only) |
| t1e1-v4e | 0.251980, 0.260470, 0.253237, 0.250132 | **0.253955** | 0.691 s | 0 | Plan resolve ~71 s (cold) |
| t1e1-v4 | 0.288330, 0.291046, 0.286378, 0.292990 | **0.289686** | 0.630 s | 0 | Plan build ~783 s; siphon wait ~599 s until plan mirrored to T |

### 2.3 FSDP8 — failed / incomplete attempts (for forensics)

| Job | Layout | Outcome | Cause |
|---|---|---|---|
| `…-004806`, `…-012929` | t1e1-v8e / v8 | exit 124 | Ray: `run_dapo.sh` `ray stop --force` killed kept cluster; actors never scheduled |
| `…-023942` | t1e1-v8e | exit 1 | MORI `pointer 0x0` — siphon registered bare FSDP params (no resident memory) |
| `…-032222` | t1e1-v8e | exit 1 | Plan built on R only; T `FileNotFoundError` on `plan.pt` (90 s wait) |
| `…-044439` | t1e1-v8 | killed mid-run | 1 transfer (0.188546); suite moved on / local `/tmp` wipe |
| `…-021146` | t1e1-v4e | no exit / empty | early aborted attempt |

Earlier misdiagnoses (reverted): Ray token auth (`RAY_AUTH_MODE=disabled`), pinning `ray_address`, `setsid ray start`. Real fix was `LUMENRL_KEEP_RAY_CLUSTER` in FSDP `run_dapo.sh`.

---

## 3. Setup / ops notes that mattered on Monaco

1. **Keep Ray cluster.** FSDP branch’s `examples/DAPO/run_dapo.sh` must skip `ray stop --force` when `LUMENRL_KEEP_RAY_CLUSTER=1` (same guard as DDP tree).
2. **Plan cache mirror.** No NFS: `plan.pt` written on R must be rsynced to T before trainer `init_source`. Continuous mirror + `LUMEN_SIPHON_PLAN_WAIT_S=900` unblocked FSDP8 after cold plan builds (~11–13 min).
3. **FSDP siphon wiring.** Siphon already had `shard_sources`; runtime needed `gather_shards` + `MegatronFsdpSide` registering Megatron-FSDP resident BF16 slices (`piece_name`). Lumen `siphon_call` must use that side under `_mfsdp`.
4. **Conductor SSH.** Host UID 12376 needs local NSS shim (`LD_PRELOAD=/tmp/nss-fake/fake_pw.so`) from the launch machine; `/tmp` wipes kill suite scripts and the shim.
5. **Timeouts.** FSDP8 setup ≈ 30–40 min before first transfer; use `RUN_TIMEOUT≥4800`.

---

## 4. What was left unfinished on Monaco

| Item | Status |
|---|---|
| FSDP8 × vLLM TP2 EP2 × 4 | not started |
| FSDP8 × vLLM TP2 × 4 | not started |
| FSDP8 × vLLM TP1 × 8 | not started |
| Like-for-like Crusoe FSDP8 (same branch/image/layouts) | **in progress** — code+scripts on NFS (`rocm-siphon-fsdp8`, `Lumen-RL-fsdp8`, `job-siphon-fsdp8.sh`); fat image `lumenrl:rocm10.3-gfx942-950-20261119` rsync resumed (~7.3/16.4 GB), auto-submits all 7 layouts when archive verifies |
| Commit of uncommitted siphon + FSDP2 worktree fixes | user commits |

---

## 5. Log locations

- Monaco host logs: `/home/sugovind/lumen-xnode/logs/run-monaco-table-*` and `run-monaco-fsdp8-*` on `10.235.225.36` (and trainer mirror of RDV on `.89`).
- Per-run: `siphon_<layout>.log`, `host.node0.log`, `*.outer`.
- Related note: `notes/monaco-debug-t4e8-v8e-findings.md`.

---

## 6. One-line bottom line

On Monaco, siphon DDP is **faster than Crusoe on 8-GPU engines and slower on 4-GPU engines**; siphon FSDP8 (TP1 EP1) **works** after shard-aware registration, lands ~0.19–0.29 s on the layouts we finished, and is **not directly comparable** to Crusoe’s older TP4 EP8 full-sharding column until Crusoe is re-run with the same FSDP8 trainer.
