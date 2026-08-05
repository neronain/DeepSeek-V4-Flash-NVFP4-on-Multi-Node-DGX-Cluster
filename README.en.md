# DeepSeek-V4-Flash NVFP4 บนคลัสเตอร์ DGX หลายเครื่อง

> Runs on **DGX Spark (GB10) × 2 เครื่อง** · vLLM (Docker, stacked)
> Tested by **neronain** · <https://www.facebook.com/neronain.minidev>

🇹🇭 ภาษาไทย: [README.md](README.md) — the Thai README is the primary document.

---

## In 30 seconds

| | |
|---|---|
| Model | `nvidia/DeepSeek-V4-Flash-NVFP4` |
| Engine | vLLM (Docker, stacked) |
| Runtime image | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` |
| Tested on | DGX Spark (GB10) × 2 เครื่อง |
| Memory needed | ~157 GB (NVFP4) แบ่ง TP=2 |
| Features | reasoning · tools · kv-cache fp8 · context ยาว |

A large MoE that still needs two machines after NVFP4 quantisation. This page focuses on what
actually prevents startup — found by running it, not from theory.

**Three things must all be right:** the image must carry DeepSeek V4 kernels; `--kv-cache-dtype fp8`
is mandatory (the `fp8_ds_mla` layout rejects `auto`); and `HF_HUB_CACHE` must match the cache
layout, or you get `LocalEntryNotFoundError` with every file present.

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

---

## Verify it actually works

```bash
curl -s http://localhost:8000/v1/models | jq .
```

With the LMDS controller: `./deepseek-v4-flash-nvfp4-stacked.sh test-text`

---

## When it breaks

| Symptom | Check |
|---|---|
| `/health` never answers | Large models take minutes to load — read the log before assuming it hung |
| OOM during warm-up | Lower `--gpu-memory-utilization` by 0.05 at a time |
| `LocalEntryNotFoundError` with all files present | HF cache is in the other layout — point `HF_HUB_CACHE` at it |
| Download stalls / files missing | `verify-files`, then `download` again (it resumes) |
| Not sure whether the weights are already here | `lmds scan` |

---

## Provenance and scope

- Based on real runs on two DGX Sparks (2026-08-05)
- `lmds recipes nvidia/DeepSeek-V4-Flash-NVFP4` already carries every value on this page

> **Status**: every value on this page comes from a real run on the hardware named above, not an
> estimate. Different hardware may need a different context length and `--gpu-memory-utilization`.

## See also

- [AutoDeployDGXProject](https://github.com/neronain/AutoDeployDGXProject) — generates controllers from a Hugging Face link
- [dgx-spark-all-controllers](https://github.com/neronain/dgx-spark-all-controllers) — 22 ready-to-run DGX Spark controllers
