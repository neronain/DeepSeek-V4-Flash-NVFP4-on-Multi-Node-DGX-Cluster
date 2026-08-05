# DeepSeek-V4-Flash NVFP4 บนคลัสเตอร์ DGX หลายเครื่อง

> รันบน **DGX Spark (GB10) × 2 เครื่อง** · vLLM (Docker, stacked)
> ทดสอบจริงโดย **neronain** · <https://www.facebook.com/neronain.minidev>

🇬🇧 English: [README.en.md](README.en.md)

---

## สรุปใน 30 วินาที

| | |
|---|---|
| โมเดล | `nvidia/DeepSeek-V4-Flash-NVFP4` |
| Engine | vLLM (Docker, stacked) |
| Runtime image | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` |
| ฮาร์ดแวร์ที่ทดสอบ | DGX Spark (GB10) × 2 เครื่อง |
| หน่วยความจำที่ต้องมี | ~157 GB (NVFP4) แบ่ง TP=2 |
| ฟีเจอร์ | reasoning · tools · kv-cache fp8 · context ยาว |

โมเดล MoE ขนาดใหญ่ที่ NVFP4 แล้วยังต้องใช้สองเครื่อง — เอกสารนี้เน้นสิ่งที่ทำให้ start ไม่ผ่าน
ซึ่งเจอจากการรันจริง ไม่ใช่ทฤษฎี

**สามอย่างที่ต้องถูกพร้อมกัน ไม่งั้นไม่ขึ้น:**

1. **image** — ต้องเป็น build ที่มี kernel ของ DeepSeek V4 (`dspark-vllm-gx10`)
   image vLLM ทั่วไปไม่มี
2. **`--kv-cache-dtype fp8`** — attention layout `fp8_ds_mla` ไม่รับ `auto`
   (`AssertionError: only supports fp8 kv-cache, got auto`)
3. **cache layout** — weight ที่โหลดด้วย HF รุ่นเก่าอยู่ที่ `$HF_HOME/models--…` ไม่ใช่ `hub/`
   ถ้าไม่ตั้ง `HF_HUB_CACHE` ให้ตรงจะได้ `LocalEntryNotFoundError` ทั้งที่ไฟล์ครบทุกไฟล์

---

## วิธีที่แนะนำ: ใช้ LMDS สร้างให้

[LMDS](https://github.com/neronain/AutoDeployDGXProject) เก็บ**สูตรของโมเดลนี้ไว้ในตัวโปรแกรมแล้ว**
— ค่า image / parser / quantization ที่หน้านี้อธิบาย ถูกใส่ให้อัตโนมัติ **โดยไม่ต้องมี API key ของ LLM**

```bash
lmds recipes nvidia/DeepSeek-V4-Flash-NVFP4          # ดูสูตร + ที่มา
lmds deploy nvidia/DeepSeek-V4-Flash-NVFP4 \
  --target dgx-spark-stacked --no-llm --yes
```

ได้ controller มาตรฐาน (ผ่าน `audit-controllers.py` 0 error 0 warning) ที่มีครบทั้ง
`download` · `verify-files` · `start` · `stop` · `status` · `logs` · `test-text` · `doctor`

### ใช้ LMDS จัดการทั้งคลัสเตอร์

```bash
lmds node cluster --write deepseek-v4-flash-nvfp4 --on spark-head
./deepseek-v4-flash-nvfp4-stacked.sh prepare-runtime
./deepseek-v4-flash-nvfp4-stacked.sh verify-files
./deepseek-v4-flash-nvfp4-stacked.sh sync-worker && ./deepseek-v4-flash-nvfp4-stacked.sh verify-worker
./deepseek-v4-flash-nvfp4-stacked.sh start --gpu-util 0.80
```

controller ตั้ง `HF_HUB_CACHE` ให้ตรงกับ layout เอง และหา NCCL interface + RoCE HCA จาก cluster IP
— ไม่ต้องย้ายไฟล์หรือกรอกชื่อ interface

---

## ตรวจว่าใช้งานได้จริง

```bash
curl -s http://localhost:8000/v1/models | jq .
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash-nvfp4","messages":[{"role":"user","content":"2+2=?"}],"max_tokens":32}'
```

ถ้าใช้ controller ของ LMDS: `./deepseek-v4-flash-nvfp4-stacked.sh test-text`

---

## พังแล้วดูตรงไหน

| อาการ | ตรวจ |
|---|---|
| `/health` ไม่ตอบสักที | โมเดลใหญ่โหลดนานหลายนาที — ดู log ก่อนสรุปว่าค้าง |
| OOM ตอน warm-up | ลด `--gpu-memory-utilization` ทีละ 0.05 |
| `LocalEntryNotFoundError` ทั้งที่ไฟล์ครบ | HF cache อยู่คนละเลย์เอาต์ — ตั้ง `HF_HUB_CACHE` ให้ตรง |
| ดาวน์โหลดค้าง/ไฟล์ไม่ครบ | `verify-files` แล้ว `download` ซ้ำ (resume ได้) |
| ไม่รู้ว่ามีโมเดลอยู่ในเครื่องแล้วหรือยัง | `lmds scan` |

| `LocalEntryNotFoundError` ทั้งที่ไฟล์ครบ | HF cache อยู่คนละ layout — `lmds scan` จะบอกว่าอยู่แบบไหน |
| `only supports fp8 kv-cache, got auto` | ไม่ได้ตั้ง `--kv-cache-dtype fp8` |
| `Mismatched number of arguments` ตอน load | JIT cache ค้างจากคนละเวอร์ชัน — `clear-fi-cache` |
| image ต่างกันระหว่างเครื่อง | `prepare-runtime` ล็อก image ให้ตรงกันทุก node |

---

## ที่มาและขอบเขต

- อ้างอิงจากการรันจริงบน DGX Spark 2 เครื่อง (5 ส.ค. 2569)
- `lmds recipes nvidia/DeepSeek-V4-Flash-NVFP4` มีค่าทั้งหมดนี้อยู่แล้ว — ไม่ต้องจำเอง

> **สถานะ**: ค่าที่ระบุในหน้านี้มาจากการรันจริงบนฮาร์ดแวร์ที่ระบุไว้ ไม่ใช่การประเมิน
> เครื่องต่างรุ่นอาจต้องปรับ context และ `--gpu-memory-utilization` เอง

## อ่านต่อ

- [AutoDeployDGXProject](https://github.com/neronain/AutoDeployDGXProject) — ตัวสร้าง controller อัตโนมัติจากลิงก์ Hugging Face
- [dgx-spark-all-controllers](https://github.com/neronain/dgx-spark-all-controllers) — controller สำเร็จรูป 22 ตัวสำหรับ DGX Spark
