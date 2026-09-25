LumenRL Weight Transfer over MORI — Gap Review and Work Plan
Owner: Wen · Status: Draft · Version: rev 1 · Date: 2026-08-27

Status note (2026-09-23). Chao's branch chao/rdma_orch adds direct trainer-to-rollout weight transfer over RDMA for ATOM, with test_atom_rdma_orchestration.py, and chao/collective_rpc_consumer drives ATOM rollout workers through collective_rpc. ATOM may therefore become a usable receiver earlier than this plan assumes; Phase 4.4 can pull forward, and the mapper interface in Phase 2 should be reviewed against Chao's orchestration design before it is fixed. The RCCL path now has substantial test coverage (tests/unit/test_rdma_weight_transfer.py, 827 lines; test_vllm_colocate_worker_ext.py, 748 lines), which gives Phase 1 a ready parity baseline. CI still runs only docs.yml, so those tests do not execute on PR.

Request. The LumenRL team asked for help using MORI for trainer-to-rollout weight transfer. ATOM's RL integration is not ready, so vLLM is the inference engine for development and validation. The MORI path is also the place to implement the transfer features the project currently lacks, instead of retrofitting them into each existing mechanism.

Code refs. LumenRL main @ 7cfec8f (re-checked 2026-09-23; was b949b72 at review time), dev/dsv4-grpo, chao/rdma_orch. Submodule third_party/mori → ZhangDanyang-AMD/mori, branch sdma-new. Related docs: lumenrl-gap-analysis.md (IDs W1–W6), megatron-te-lowprecision-plan.md.

1. Starting point
1.1 MORI is not used for transport in LumenRL today
lumenrl/transfer/mori_io_store.py does not use the MORI library. It implements intra-node HIP IPC peer-to-peer copy: each process allocates a GPU staging buffer, writes the storage IPC handle to a pickle file under /dev/shm/mori_io, the peer reconstructs the tensor from that handle, data moves by device-to-device copy, and completion is signalled through files in /dev/shm/mori_io_ready. There is no import mori anywhere under lumenrl/. Re-verified 2026-09-23 across main, dev/dsv4-grpo, dev/dapo_release and chao/rdma_orch: still no MORI transport for weights on any branch. The third_party/mori submodule is pinned but not referenced by LumenRL Python code.

Implications:

The SDDD "Mooncake or MORI" hidden-state choice is Mooncake or HIP IPC. The IPC option works only within a node.
MORI weight transfer is new work, with no existing integration to extend.
The only MORI usage in the wider stack is mori.ops for MoE expert-parallel dispatch (forward-only in Lumen, training-grade in the ROCm Megatron-LM fork). That is a different MORI component from the transfer engine.
The file and class names should be corrected so that nobody plans against a MORI-IO integration that does not exist.
1.2 Existing weight-transfer mechanisms
Mechanism	Scope	What it has	What it lacks
RCCL broadcast over RoCE GDRDMA	Cross-node, production path in the two-node guide	Persistent 9-rank group (1 sender + all vLLM TP workers); per-bucket header [command, metadata_bytes, payload_bytes, version] with version check; JSON manifest broadcast in-band; flat payload with zero-copy receive views; verify_full_load coverage check; reset_prefix_cache after load; telemetry (bytes, buckets, seconds, Gb/s)	Every TP worker receives the full weight set; fixed group membership; no integrity fingerprints; no atomic activation; per-bucket allocation on both sides
ZMQ + HIP IPC bucketed (vendored from verl)	Colocated, same GPU	Shared 512 MB staging buffer, REQ/REP flow control, ROCm device-ordinal patch	No versioning, no manifest verification, per-call buffer allocation and empty_cache, no double-buffering, TP > 1 wiring unclear, BF16 only
Filesystem safetensors	Fallback	Works without any network setup	Defaults to node-local /dev/shm while auto-selected for disaggregated topologies; no atomic rename; version file written after tensors
ATOM snapshot reload	ATOM subprocess wrapper	—	Full reload per step; ATOM fork's native sleep/wake and SHM in-place load are not on the production path
Ray object store	Covered by tests	—	Contains an unscaled .to(float8_e4m3fn) cast and an unsafe aggregate="mean" option
FP8 per-block quantize-in-stream (now on main)	RCCL path only	128×128 blocks, continuous FP32 inverse scales, vLLM fp8_per_block-compatible, model-aware skip list; streaming PP broadcast	Merged to main as lumenrl/engine/training/fp8_weight_quantizer.py with unit tests; still one format dialect and tied to one transport
weight_integrity.py on the dsv4 branch provides per-tensor fingerprints and FNUZ-aware NaN/Inf statistics, but no transport uses it.

1.3 Gaps the MORI path can close
ID	Gap	How the MORI path addresses it
W1	Six mechanisms with no shared framing	Extract version header, manifest, coverage check and telemetry into a transport-independent layer; MORI becomes its second implementation after RCCL
W4a	Redundant transmission: each TP worker gets the full model	Targeted per-shard transfer
W4b	Fixed group membership	Point-to-point transfer has no fixed collective group
W4c	No transfer-engine tier	This work provides it
—	Quantized payload on one branch and one transport	Move the quantizer into the shared layer with format dispatch
—	Integrity checks unused	Fingerprints in the manifest, verified on receive
W6	No delta payload (verl, slime, Miles have one)	Payload option on the same transport, after MX-format delta density is measured
W5	verl Checkpoint Engine transport layer is extensible and has no AMD backend	The same MORI backend becomes a verl transport plugin
Out of scope here: ATOM receiver work (tracked by the ATOM integration list), and the two ROCm platform bugs under sleep/wake (W3), except where registration lifecycle depends on them.

2. Design decisions
2.1 Transfer architecture
MORI in place of the broadcast. One sender, all receivers, full payload into a staging buffer, existing load_weights path unchanged. Gains dynamic membership and removes the rendezvous coupling of a collective group. Low risk; modest bandwidth gain.
Targeted per-shard transfer. The sender computes, for each receiver, the bytes that receiver needs and writes only those. Removes the TP-fold redundancy. Requires a mapper from parameter name to receiver and byte range, which needs the receiver's parallelism layout.
Source-side engine replica. A host-resident copy runs the inference engine's own loader so destination layout is produced by engine code, then bytes move peer to peer. This is the shape used by Miles with Mooncake. Layout mapping is handled by construction; cost is host memory and an extra copy.
Plan: build option 1 as the vehicle, then move to option 2. Keep option 3 available if the mapper becomes expensive to maintain across models.

2.2 Write target on the receiver
Staging buffer, then load_weights. vLLM's loader keeps doing QKV fusion, gate/up fusion, expert stacking, TP sharding and quantization handling. One extra device copy.
Direct into parameter storage. Zero copy, but the sender must produce tensors already fused, sharded and quantized to match the receiver's parameter layout exactly.
Plan: staging in Phase 1. In Phase 2, direct writes for parameters whose layout already matches, staging for the rest.

2.3 Push or pull
Default to one-sided writes from the trainer, since the trainer holds the source data and knows when a version is complete. Revisit if receiver-driven pull simplifies partial-failure recovery.

2.4 Control plane
The current IPC store exchanges handles through files in /dev/shm, which only works within a node. Endpoint and memory descriptors should be exchanged over Ray (already present) so the same code runs across nodes.

3. Work plan
Phase 0 — Measure and decide (1–2 weeks)
Task	Detail
0.1	Weight-shaped MORI-IO microbenchmark. Tensor-size distributions taken from Qwen3-30B-A3B and DSV4. Intra-node and inter-node. Measure registration cost per buffer, bucketed versus per-tensor registration, and one-to-many fan-out.
0.2	Baseline the current RCCL path on the two-node configuration using its existing telemetry.
0.3	Choose architecture (2.1) and write target (2.2) from 0.1 and 0.2.
0.4	Move third_party/mori to an upstream ROCm branch or tag.
0.5	Rename or document mori_io_store.py and mori_hidden_states_connector.py as HIP IPC.
Exit: an estimate of the TP degree and node count at which targeted transfer beats broadcast, and a registration strategy that keeps registration off the per-step critical path.

Phase 1 — MORI transport at feature parity, validated on vLLM (3–4 weeks)
Task	Detail
1.1	Transport abstraction. Move framing (version header, manifest, payload layout, coverage verification, telemetry) out of the RCCL path into a layer with send_bucket / recv_bucket implemented by each backend. RCCL keeps working unchanged through it.
1.2	Control plane over Ray for endpoint and memory-descriptor exchange.
1.3	MORI backend: pre-registered staging arenas on sender and receivers, bucketed writes, completion signalling, timeouts, and defined behaviour when the sender fails mid-stream.
1.4	vLLM receiver as a worker extension driven by collective_rpc: register staging buffer, receive, pass to load_weights, run coverage verification, call reset_prefix_cache.
1.5	Registration lifecycle across vLLM sleep and wake. Level-2 sleep discards weight memory, so registrations must be re-established on wake. Measure the cost.
Exit: the MORI path passes the same smoke test and 200-step run as the documented RCCL flow, and generation matches a frozen RCCL baseline.

Phase 2 — Targeted transfer (3–4 weeks)
Task	Detail
2.1	Receiver parallelism info through collective_rpc: TP rank, shard ranges, fused-parameter layout.
2.2	Sender-side plan: parameter name → (receiver, byte range). Start with Qwen3 and Qwen3-MoE; keep the mapper interface engine-neutral so ATOM can implement it later.
2.3	Sender-side shard extraction so each worker receives only its slice.
2.4	Direct-to-parameter writes where the layout matches; staging elsewhere.
Exit: bytes on the wire reduced by approximately the TP factor, with measured wall-time improvement over Phase 1 and over the RCCL baseline.

Phase 3 — Payload features (overlaps Phase 2)
Task	Detail
3.1	Port FP8 per-block quantize-in-stream from dev/dsv4-grpo into the shared layer.
3.2	Format dispatch keyed on the receiver's quantization config, per tensor group. Entries: FP8 128-block with continuous FP32 scales (Qwen, GLM dialect), FP8 128-block with UE8M0 scales (DeepSeek V3.1+ dialect), MXFP4 g32 E8M0 (K3, GPT-OSS, DSV4 experts).
3.3	Integrity fingerprints from weight_integrity.py included in the manifest and checked on receive.
3.4	Atomic activation: serving weights switch only after the final bucket arrives and verification passes, so a failure mid-stream cannot leave a mix of old and new parameters.
3.5	Delta payload, after measuring changed-byte density for MXFP8 and MXFP4. FP8 delta is known to stay sparse when diffed after quantization; NVFP4 is known not to; the MX formats are unmeasured.
Phase 4 — Scale-out and consolidation
Task	Detail
4.1	Multiple replicas with dynamic membership: engines join or leave without rebuilding a group.
4.2	Broadcast versus targeted-transfer crossover measured across node counts and TP degrees.
4.3	Remove the Ray object-store path. Restrict the filesystem fallback to single node and add atomic rename. Keep the IPC path for colocated runs but move it onto the shared framing.
4.4	ATOM as a receiver once its RPC and RDMA items land, reusing the same transport and mapper interface.
Phase 5 — Upstream
Task	Detail
5.1	MORI-IO transport for verl's Checkpoint Engine.
5.2	Propose the receiver-side API (parallelism info, buffer registration, post-apply hook) to vLLM so the worker extension does not remain a local patch.
4. Validation
vLLM is used because ATOM is not ready. Building against vLLM's public extension points also makes the result usable by verl and vime on ROCm. The mapper and receiver interfaces should not assume vLLM internals, so that ATOM becomes a second implementation.

Checks, each intended to run in CI once CI exists:

Check	Pass condition
Fingerprint round-trip	Sender and receiver fingerprints match for every tensor
Coverage	Every receiver parameter loaded; fail on any missing parameter
Generation against frozen baseline	Per-token log-prob difference against the RCCL path within threshold on max and p99.9; bitwise equality expected for BF16 payload
Multi-step RL	Same seed, RCCL versus MORI, matching reward and KL curves
Failure injection	Sender killed mid-stream: receiver keeps serving the previous version and reports the failure
Sleep/wake soak	Repeated sleep → register → receive → wake cycles without memory growth or registration leaks
5. Risks
Risk	Mitigation
Registration cost dominates for many small tensors	Pre-registered arenas and bucketing, measured in Phase 0 before committing
MORI-IO tuned for KV-block transfer rather than this access pattern	Phase 0 benchmark uses weight-shaped size distributions
No upstream vLLM API for exposing parameter addresses	Worker extension first, upstream proposal in Phase 5
Mapper is model-specific (fused QKV, gate/up, stacked experts)	Start with two model families; fall back to the source-side replica design if maintenance cost grows
Registration invalidated by sleep/wake; interacts with the known ROCm wake page fault	Tie registration to wake explicitly; include the soak test above
Submodule pinned to a personal-fork branch	Phase 0 task 0.4
6. Open questions
What is MORI-IO's registration cost per buffer on MI308X and MI355X, and does it support registering a large arena once and addressing sub-ranges?
Does MORI-IO expose completion notification suitable for many small writes, or is per-bucket completion needed?
Which vLLM version does LumenRL pin, and is collective_rpc with worker extensions available there?
Should the SDDD hidden-state path move onto the same MORI transport for cross-node use, since its current IPC path is intra-node only?
Who owns the receiver-side mapper for ATOM once ATOM is ready?
Changelog
2026-09-23 (rev 2) — Repo re-checked against main 7cfec8f. Premise confirmed: no MORI transport for weights on any branch. FP8 quantize-in-stream is now on main rather than the dsv4 branch. Added status note on chao/rdma_orch (ATOM RDMA weight transfer) and chao/collective_rpc_consumer, and on the new RCCL transfer tests as a Phase 1 parity baseline.

2026-08-27 (rev 1) — Initial plan. Records that mori_io_store.py is HIP IPC and MORI is not used for transport in LumenRL. Maps W1, W4, W5, W6 onto a MORI transport validated on vLLM, in five phases with exit criteria.