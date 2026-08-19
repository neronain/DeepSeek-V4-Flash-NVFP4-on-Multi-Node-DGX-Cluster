# DeepSeek-V4-Flash NVFP4 on a Multi-Node DGX Cluster

> Runs on **DGX Spark (GB10) × 2** · vLLM (Docker, stacked)
> Tested by **neronain** · <https://www.facebook.com/neronain.minidev>

🇹🇭 ภาษาไทย: [README.md](README.md) — the Thai README is the primary document.

---

## In 30 seconds

| | |
|---|---|
| Model | `nvidia/DeepSeek-V4-Flash-NVFP4` |
| Engine | vLLM (Docker, stacked) |
| Runtime image | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` |
| Tested on | DGX Spark (GB10) × 2 |
| Memory needed | ~157 GB (NVFP4), split across TP=2 |
| Features | reasoning · tools · fp8 kv-cache · long context |

A large MoE that still needs two machines after NVFP4 quantisation. This page focuses on what
actually prevents startup — found by running it, not from theory.

**Three things must all be right, or it will not start:**

1. **The image** must carry the DeepSeek-V4 kernels (`dspark-vllm-gx10`). A stock vLLM image does not.
2. **`--kv-cache-dtype fp8`** is mandatory — the `fp8_ds_mla` attention layout rejects `auto`
   (`AssertionError: only supports fp8 kv-cache, got auto`).
3. **`HF_HUB_CACHE`** must match the on-disk cache layout. Weights fetched by an older `huggingface_hub`
   live under `$HF_HOME/models--…` rather than `hub/`; point at the wrong one and you get a
   `LocalEntryNotFoundError` with every file present.

---

## Recommended: let LMDS generate it

[LMDS](https://github.com/neronain/AutoDeployDGXProject) ships **a recipe for this model**, so the
image, parser and quantisation settings described here are filled in automatically — **no LLM API
key required**.

```bash
lmds recipes nvidia/DeepSeek-V4-Flash-NVFP4          # the recipe and where it came from
lmds deploy nvidia/DeepSeek-V4-Flash-NVFP4 \
  --target dgx-spark-stacked --no-llm --yes
```

You get a standard controller (0 errors, 0 warnings from `audit-controllers.py`) with
`download` · `verify-files` · `start` · `stop` · `status` · `logs` · `test-text` · `doctor`.

### Manage the whole cluster with LMDS

```bash
lmds node cluster --write deepseek-v4-flash-nvfp4 --on spark-head
./deepseek-v4-flash-nvfp4-stacked.sh prepare-runtime
./deepseek-v4-flash-nvfp4-stacked.sh verify-files
./deepseek-v4-flash-nvfp4-stacked.sh sync-worker && ./deepseek-v4-flash-nvfp4-stacked.sh verify-worker
./deepseek-v4-flash-nvfp4-stacked.sh start --gpu-util 0.80
```

The controller sets `HF_HUB_CACHE` to match the cache layout itself, and finds the NCCL interface and
RoCE HCA from the cluster IP — no moving files, no typing interface names.

---

## Verify it actually works

```bash
curl -s http://localhost:8000/v1/models | jq .
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash-nvfp4","messages":[{"role":"user","content":"2+2=?"}],"max_tokens":32}'
```

With the LMDS controller: `./deepseek-v4-flash-nvfp4-stacked.sh test-text`

---

## When it breaks

| Symptom | Check |
|---|---|
| `/health` never answers | Large models take minutes to load — read the log before assuming it hung |
| OOM during warm-up | Lower `--gpu-memory-utilization` by 0.05 at a time |
| `LocalEntryNotFoundError` with all files present | HF cache is in the other layout — point `HF_HUB_CACHE` at it (`lmds scan` tells you which) |
| `only supports fp8 kv-cache, got auto` | `--kv-cache-dtype fp8` is not set |
| `Mismatched number of arguments` on load | A JIT cache from a different build is stale — `clear-fi-cache` |
| Image differs between the two nodes | `prepare-runtime` locks the image to the same ID on every node |
| Download stalls / files missing | `verify-files`, then run `download` again (it resumes) |
| Not sure whether the weights are already here | `lmds scan` |

---

## Provenance and scope

- Based on real runs on two DGX Sparks (2026-08-05).
- `lmds recipes nvidia/DeepSeek-V4-Flash-NVFP4` already carries every value on this page.

> **Status**: every value on this page comes from a real run on the hardware named above, not an
> estimate. Different hardware may need a different context length and `--gpu-memory-utilization`.

## See also

- [AutoDeployDGXProject](https://github.com/neronain/AutoDeployDGXProject) — generates controllers from a Hugging Face link
- [dgx-spark-all-controllers](https://github.com/neronain/dgx-spark-all-controllers) — ready-to-run DGX Spark controllers
