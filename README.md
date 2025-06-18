🎮💬 Valorant Game Chatbot — Fine-Tuned with LoRA

📌 **Overview**

This project fine-tunes a large language model (LLM) for **Valorant game-related conversations**, using:
- ✅ A base LLM (`microsoft/phi-2` — or any other you choose)
- ✅ Lightweight **LoRA** (Low-Rank Adaptation) for efficient fine-tuning
- ✅ Custom game dataset in JSON format
- ✅ Hugging Face `Trainer` for training
- ✅ A deployable chatbot pipeline for real-time Q&A

---

⚡ **Tech Stack**

- **Transformers** — LLM and training utilities
- **Datasets** — to load and format custom JSON data
- **PEFT (LoRA)** — for parameter-efficient fine-tuning
- **BitsAndBytes** — for 4-bit or 8-bit quantization to save GPU memory
- **PyTorch** — core backend
- **Hugging Face Hub** — to push your trained model for reusability

---

## 🚀 **How it works**

### 1️⃣ **Install Dependencies**

bash
pip install torch transformers sentencepiece accelerate datasets peft bitsandbytes
