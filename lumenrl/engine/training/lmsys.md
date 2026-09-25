The post compares two ways of getting new weights from a Miles trainer (Megatron) to SGLang rollout engines on separate GPUs. The default way gathers each weight into one full copy and broadcasts that copy to every inference GPU over NCCL. Each GPU then keeps only its own slice. The proposal gives each trainer GPU a copy of one inference GPU's slice of the model, held in CPU memory. That copy is filled in the exact byte layout SGLang uses. Each trainer GPU then writes only the bytes each inference GPU needs, straight into that GPU's weight memory, using RDMA (Mooncake TransferEngine). For 1T-parameter Kimi-K2, this cuts the update from about 53 s to about 7.2 s.

## The default workflow: NCCL broadcast

Some background. The trainer splits the model across GPUs in three ways:

- **Pipeline parallelism (PP):** different groups of layers live on different GPUs.
- **Tensor parallelism (TP):** individual weight matrices are split across GPUs.
- **Expert parallelism (EP):** in mixture-of-experts models, different experts live on different GPUs.

SGLang splits the model its own way, usually with a different TP/EP layout.

After each training step, the default flow runs once per pipeline stage, bucket by bucket:

1. **Gather on the trainer.** Inside each pipeline stage, the GPUs all-gather across TP and EP. The "head rank" of that stage (one GPU) ends up holding the full weight tensors in Hugging Face format.
2. **Broadcast.** The head rank broadcasts those full tensors over an NCCL group that includes every SGLang GPU (`update_weights_from_distributed`).
3. **Load and discard.** Each SGLang GPU runs its normal `load_weights`: it slices out its own shard, fuses tensors (such as QKV), quantizes, and throws the rest away.

They name three problems with this:

- **Redundancy.** Every inference GPU receives the whole tensor even though it keeps only a slice. With expert parallelism of size `ep`, each GPU receives about `ep × P` bytes but needs only `P`, where `P` is the number of parameters one inference GPU actually holds.
- **Inactivity.** Only one head GPU per pipeline stage sends. Every other trainer GPU and its network card sits idle.
- **Rigidity.** NCCL groups have fixed membership. Adding or removing an inference engine means rebuilding the group. Every rank must also call the collective in lockstep, so one slow receiver stalls all of them.

Meanwhile, both training and rollout are stopped for the whole update.

## What they propose: point-to-point RDMA with a CPU copy of the engine

**Setup, done once:**

1. `get_remote_instance_transfer_engine_info`: each SGLang GPU registers its weight memory with the NIC and reports the addresses.
2. `get_parallelism_info`: SGLang reports its TP/EP layout.
3. `build_transfer_plan`: decide which trainer GPU sends to which inference GPU.
4. `create_engine_replica`: build a copy of one SGLang GPU's slice of the model in the trainer's CPU memory, and register it for RDMA.

**Each update:**

1. **Pause** SGLang.
2. **Gather.** Same TP/EP all-gather as before. Every trainer GPU in a pipeline stage now holds the full tensors, not just a head rank.
3. **Load into the CPU copy.** Run SGLang's own weight loader on the CPU copy, using the target GPU's parallelism settings. The copy now holds exactly the bytes that target GPU should have: sliced, fused and laid out correctly.
4. **RDMA write** those bytes into the target GPU's registered weight memory. The receiver's CPU isn't involved, and no staging buffer is needed on the receiver.
5. **Reuse the copy for the next target.** All SGLang GPUs are built the same way, so one CPU copy is reloaded for each target GPU in turn. The write to one target must finish before the copy is overwritten for the next. The last write runs in a background thread pool, so the trainer can move on to the next bucket.
6. **Hold incomplete tensors.** One SGLang tensor can come from several Hugging Face tensors, for example `q_proj`, `k_proj` and `v_proj` all go into `qkv_proj`. Partial pieces wait in a side buffer until every piece has arrived. Non-expert weights are sent first, then expert weights.
7. **Post-process** on the SGLang GPUs, for work the CPU copy can't do: quantization and derived tensors such as DeepSeek MLA's `w_kc` / `w_vc`.
8. **Update the weight version** and **resume generation**.

**Assigning senders to receivers** uses round-robin. Their example has 32 trainer GPUs split into 4 pipeline stages, and 2 SGLang engines with 16 GPUs each. In pipeline stage 0, trainer GPUs 0–7 each take one of SGLang GPUs 0–7, then GPUs 8–15. GPU 16 in the second engine is identical to GPU 0 in the first, so it goes to the same sender. Sender 0 ends up with targets {0, 16, 8, 24}. Every inference GPU receives from every pipeline stage.

## The trade-off

This is their cost table, where M is the number of trainer GPUs and K is the gather bucket size:

| | NCCL broadcast | RDMA P2P |
|---|---|---|
| Trainer GPUs sending | one per pipeline stage | all M |
| Bytes received per inference GPU | `ep × P` | `P` |
| Extra memory on the trainer | K | about K plus a P-sized CPU copy |
| Extra memory on the inference GPU | K | 0 |

In short, they spend CPU memory on the trainer side to avoid sending redundant bytes. For Kimi-K2 that's about 32 GB per trainer GPU. In exchange, every trainer GPU and its NIC is used, receivers get only what they need, and engines can join or leave without rebuilding a group.

## Why the copy is on CPU, not GPU

They first tried a GPU copy. Registering it with the NIC took tens of seconds, longer than the transfer itself, and it also took training memory. A CPU copy can be registered once and kept. Running SGLang's own loader on it also means they never hand-write layout mappings, which is how they support every model and quantization SGLang supports.

## Results

On H100 nodes with InfiniBand, the gain grows with scale and expert parallelism:

| Model | Setup | Speedup |
|---|---|---|
| GLM4-9B | 1 node | 0.98× (no gain) |
| GLM4-MoE, small EP | 1 node | 0.59× (slower) |
| Qwen3-235B | 8 nodes | 3.4× |
| Kimi-K2 1T | 32 nodes | 7.4× |

At small scale, loading into the CPU copy costs more than the network time it saves.

## Where it sits among alternatives

- **Disk reload:** minutes.
- **NCCL broadcast:** about 50 s.
- **LMSYS RDMA P2P:** about 7 s. Works with any model through SGLang's loader.
- **Perplexity's fabric-lib:** about 1.2 s. Writes from trainer GPUs directly into inference GPUs, with no CPU copy. It supports only FSDP2 and requires hand-built layout mappings.

That last row is the direct-write design we discussed. The LMSYS post chooses generality and simpler engineering over the last factor of about 6× in speed.

IPC stands for **inter-process communication**: any way for two separate processes to exchange data. Examples include pipes, sockets, shared memory, and files in `/dev/shm`. Processes can't read each other's memory by default, so they need one of these channels.

In this code, "IPC" almost always means **CUDA IPC**, which is also called HIP IPC on AMD. It's a GPU-specific kind of IPC.

## CUDA/HIP IPC

A GPU buffer allocated by one process is normally visible only to that process. CUDA IPC lets that process export a **handle** to the buffer: a small token, not the data itself. A second process on the same machine opens the handle and gets a pointer to the **same GPU memory**. Nothing is copied. Both processes now read and write the same bytes.

In this repo that looks like:

- **Sender (the trainer actor):** allocates a staging buffer on the GPU, then calls `torch.multiprocessing.reductions.reduce_tensor(buffer)` to get the handle (`bucketed_weight_transfer.py`).
- **Handle exchange:** the handle is a few bytes of metadata, sent over a **ZMQ** socket. That's what the `ipc:///tmp/lumen-colocate-zmq-...sock` paths are: Unix-domain sockets, another form of IPC, used only to pass handles and "bucket ready" messages.
- **Receiver (vLLM or ATOM worker):** calls `rebuild_ipc(handle)` and gets a tensor backed by the sender's buffer. It reads the weights straight out of that buffer and runs `load_weights`.
- **Loop:** the sender fills the buffer with the next bucket of weights, sends "ready" over ZMQ, and the receiver loads it and replies. This repeats until all weights are sent.

## Why the function at line 772 cares

CUDA IPC only works **inside one machine**, and in practice it's used here when trainer and rollout share the same GPUs. Each trainer actor needs a receiver on its own GPU to share a buffer with. `_ipc_endpoints_match_actors` checks that pairing:

- **vLLM** opens one socket per tensor-parallel worker, so any layout where `replicas × TP == actors` pairs up one-to-one.
- **ATOM** opens a single socket per replica (hardcoded `rank-0`), so only TP=1 pairs up.

If the pairing fails, or rollout runs on other GPUs or nodes, IPC can't be used. The trainer then falls back to writing safetensors files.

## How it relates to the other transports

- **IPC:** same machine, shared GPU memory, zero copy. Only for colocated setups.
- **RCCL broadcast:** GPUs send data over a collective, within a node or across nodes over RDMA.
- **MORI-IO / RDMA:** one GPU reads or writes another GPU's registered memory over the network, across nodes.

`mori_io_store.py` is also IPC under the hood. It stores HIP IPC handles as pickle files in `/dev/shm`, which is why it only works within one node despite its name.