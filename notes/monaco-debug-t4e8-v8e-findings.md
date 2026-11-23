# Monaco hang debug: siphon:t4e8-v8e (2026-11-22)

## Result
Job **completed** with CACHE_EVERY=0 (RDV-first, no plan-cache rsync during mediate).

| step | weight_sync_siphon_transfer_s |
|------|-------------------------------|
| 1 | 0.160319 |
| 2 | 0.160294 |
| 3 | 0.159806 |
| 4 | 0.160861 |
| **mean** | **0.160320** |

Matches prior Monaco MORI v8e ≈ 0.160.

## Root cause of prior "hangs"
Not siphon/MORI. Launch mediation was starved by Conductor SSH:

1. **`mirror-plan-cache.sh`** had been looping since Nov 20 doing **4× rsync every 2s** through Conductor → saturated sshd (`tilde_expand: No such uid 12376`, auth hangs).
2. **`launch-monaco-smoke.sh`** also called `sync_plan_cache` (another 4× rsync) **every mediate iteration**, competing with RDV scp of `.rdv.head` / `.rdv.worker`.
3. Batched remote prep used `pkill -f node-siphon-monaco` inside a cmdline that itself contained that string → **killed its own SSH session**.

## Fix applied in this debug run
- Killed continuous mirror.
- Used SSH ControlMaster.
- `CACHE_EVERY=0` → mediate RDV only.
- Fixed pkill self-match (`[n]ode-siphon-monaco`).

## Follow-ups for table suite
- Do **not** restart `mirror-plan-cache.sh` as a tight loop.
- Smoke launcher: ControlMaster + mediate RDV first; sync plan cache sparsely (or once at seed).
- Keep `pkill -f '[n]ode-siphon-monaco.sh'` pattern.
