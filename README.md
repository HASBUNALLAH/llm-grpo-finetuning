# 🧠 GRPO Fine-Tuning for Mathematical Reasoning

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![Library](https://img.shields.io/badge/Library-HuggingFace_TRL-yellow?style=for-the-badge)
![Method](https://img.shields.io/badge/Method-GRPO_RLHF-red?style=for-the-badge)

## 📌 Project Overview
This project implements **Group Relative Policy Optimization (GRPO)** to fine-tune the **Qwen-2.5** Large Language Model (LLM) for advanced mathematical reasoning.

Unlike traditional Supervised Fine-Tuning (SFT), GRPO uses **Reinforcement Learning** to incentivize the model to "think" before answering. We reward the model for:
1.  **Format Adherence:** Using structured `<think>` tags.
2.  **Accuracy:** Arriving at the correct numerical result (verified against the GSM8K dataset).

## 🛠 Tech Stack
* **Model:** Qwen-2.5-Instruct (Optimized with LoRA adapters)
* **Library:** Hugging Face `trl` & `peft`
* **Dataset:** GSM8K (Grade School Math 8K)
* **Hardware:** Optimized for Single-GPU training (T4/A100)

## 🚀 How to Run
### 1. Install Dependencies
```bash
pip install -r requirements.txt
