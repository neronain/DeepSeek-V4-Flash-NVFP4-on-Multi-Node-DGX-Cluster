---

# 🚀 Deploying DeepSeek-V4-Flash-NVFP4 on Multi-Node DGX Cluster (Grace Blackwell)

บทความและคู่มือนี้อธิบายขั้นตอนการ Deploy โมเดล **DeepSeek-V4-Flash-NVFP4** แบบ Multi-Node ผ่านการทำงานร่วมกันของ **SGLang** และ **Ray Cluster** โดยใช้เครื่องเซิร์ฟเวอร์ NVIDIA DGX สถาปัตยกรรม Grace Blackwell (GB10) จำนวน 2 เครื่อง เชื่อมต่อกันผ่านเครือข่ายความเร็วสูง RoCEv2 (RDMA)

📌 1. System Architecture & Hardware

* **Model:** `nvidia/DeepSeek-V4-Flash-NVFP4` (ใช้ Data Type NVFP4 ซึ่งรีดประสิทธิภาพสูงสุดบนชิปตระกูล Blackwell)
* **Engine:** SGLang (High-performance inference engine) รองรับ TensorRT-LLM และ FlashInfer
* **Cluster Manager:** Ray
* **Hardware:** 2x NVIDIA DGX (Grace Blackwell - GB10) - *1 GPU / Node*
* **Networking:** RoCEv2 (InfiniBand/RDMA) เพื่อให้กระบวนการ Tensor Parallelism (TP) ข้ามเครื่องทำงานได้โดยไม่มีคอขวด (Zero-copy network)

---

## 🛠️ 2. Prerequisites & Network Setup

ก่อนเริ่มการรันคอนเทนเนอร์ ต้องตรวจสอบการตั้งค่าเครือข่ายและสิทธิ์การใช้งานดังนี้:

1. **High-Speed Network (RoCEv2):** ตรวจสอบ Interfaces สำหรับการคุยกันข้ามโหนด (ในที่นี้คือ `enp1s0f1np1`)
2. **Master Node IP:** `10.100.152.1`
3. **Worker Node IP:** `10.100.152.2`
4. **Hugging Face Token:** จำเป็นสำหรับการโหลดโมเดล (ตั้งเป็น Environment Variable `.env` หรือส่งตรงผ่าน Docker Compose)

---

## 🐳 3. Docker Compose Configuration

สร้างไฟล์ `docker-compose.yml` สำหรับทั้ง 2 โหนด โดยหัวใจสำคัญคือการส่งต่อ Environment Variables ของ `NCCL` (NVIDIA Collective Communications Library) เพื่อให้ GPU คุยกันข้ามเครือข่ายได้

### Master Node (`10.100.152.1`)

```yaml
services:
  sglang-master:
    image: lmsysorg/sglang:latest
    container_name: sglang-master
    network_mode: "host"
    ipc: "host"
    privileged: true
    environment:
      - NCCL_SOCKET_IFNAME=enp1s0f1np1,enP2p1s0f1np1
      - NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1
      - NCCL_IB_DISABLE=0
      - NCCL_NET_GDR_LEVEL=2
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
    volumes:
      - /home/neronain/.cache/huggingface:/root/.cache/huggingface
      - /dev/shm:/dev/shm
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['all']
              capabilities: [gpu]
    command: >
      bash -c "pip install 'ray[default]' && ray start --head --port=6379 && sleep infinity"

```

### Worker Node (`10.100.152.2`)

```yaml
services:
  sglang-worker:
    image: lmsysorg/sglang:latest
    container_name: sglang-worker
    network_mode: "host"
    ipc: "host"
    privileged: true
    environment:
      # ใช้การตั้งค่า NCCL เหมือน Master Node
      - NCCL_SOCKET_IFNAME=enp1s0f1np1,enP2p1s0f1np1
      - NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1
      - NCCL_IB_DISABLE=0
      - NCCL_NET_GDR_LEVEL=2
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
    volumes:
      - /home/neronain/.cache/huggingface:/root/.cache/huggingface
      - /dev/shm:/dev/shm
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['all']
              capabilities: [gpu]
    command: >
      bash -c "pip install 'ray[default]' && ray start --address='10.100.152.1:6379' && sleep infinity"

```

---

## 🚀 4. Deployment Steps

### Step 4.1: Start Ray Cluster

รัน Docker Compose ในแต่ละเครื่องเพื่อสร้าง Ray Cluster เบื้องหลัง

```bash
# รันที่ Master และ Worker
sudo docker compose up -d

```

ตรวจสอบสถานะ Cluster (รันที่ Master):

```bash
sudo docker exec -it sglang-master ray status

```

*ผลลัพธ์ที่คาดหวัง: จำนวน Node ที่ Active ต้องเท่ากับ 2 และมองเห็นทรัพยากร GPU รวมกันทั้งหมด (Total Usage: 0.0/2.0 GPU)*

### Step 4.2: Launch DeepSeek-V4 via SGLang

เมื่อ Cluster พร้อม ให้สั่งรันเซิร์ฟเวอร์จาก Master Node โดยใช้คำสั่ง `sglang serve` พร้อมกำหนด Tensor Parallelism (`--tp 2`) และบอกให้ระบบรู้ว่าเรากระจายงานข้าม 2 โหนด (`--nnodes 2`)

```bash
sudo docker exec -it sglang-master bash -c "sglang serve \
  --model-path nvidia/DeepSeek-V4-Flash-NVFP4 \
  --tp 2 \
  --use-ray \
  --nnodes 2 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8080"

```

*ระบบจะทำการดาวน์โหลด Weights ไปเก็บไว้ที่ `/root/.cache/huggingface` และกระจายโหลดไปยัง GPU ทั้ง 2 เครื่องผ่าน NCCL*

---

## 🧪 5. Testing the Inference API

เมื่อเซิร์ฟเวอร์รันสำเร็จ (ขึ้นข้อความ `Uvicorn running on [http://0.0.0.0:8080](http://0.0.0.0:8080)`) สามารถทดสอบยิง API ได้ตามมาตรฐาน OpenAI:

```bash
curl http://10.100.152.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "What are the advantages of Grace Blackwell architecture?",
    "sampling_params": {
      "max_new_tokens": 512,
      "temperature": 0.7
    }
  }'

```

---

## 📚 6. Reference & Documentation

* **DeepSeek-V4 on Hugging Face:** [nvidia/DeepSeek-V4-Flash-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4-Flash-NVFP4)
* **SGLang GitHub Repository:** [lmsysorg/sglang](https://github.com/sgl-project/sglang)
* **NVIDIA SGLang Playbooks:** [DGX Spark Playbooks - SGLang](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/sglang)
* **Multi-Node Spark Connection Guide:** [Connect Two Sparks](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/connect-two-sparks)
* **NVIDIA NCCL Documentation:** [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)

---
