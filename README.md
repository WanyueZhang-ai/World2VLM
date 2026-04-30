# 🌍 World2VLM: Distilling World Model Imagination into VLMs for Dynamic Spatial Reasoning

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2604.26934)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/WanyueZhang/World2VLM)
[![Status](https://img.shields.io/badge/Code-Partial%20Release-blue)]()

> 🚀 Official repository for **World2VLM**, a framework that distills *world model imagination* into Vision-Language Models (VLMs) for **dynamic spatial reasoning**.

---

## 📖 Overview

**World2VLM** introduces a novel paradigm that leverages **world models** to generate imagined trajectories and distills them into VLMs for enhanced spatial understanding.

This repository contains the **core codebase** used in the paper, which is **sufficient to reproduce the main experiments**.  
A **fully complete version** of the codebase (including additional engineering components) will be released soon.

---

## ✨ Key Features

- 🌐 **World Model Simulation** for trajectory generation  
- 🎯 **Bidirectional Spatial Supervision** (A1–A4, D1–D4 tasks)  
- 🔄 **Two-stage Post-training Pipeline** (SFT + GRPO)  
- 📦 Modular and reproducible pipeline aligned with the paper  

---

## 🔗 Resources

- 📄 Paper: https://arxiv.org/abs/2604.26934  
- 🤗 Dataset: https://huggingface.co/datasets/WanyueZhang/World2VLM  

---

## 🧩 Pipeline Overview

The repository follows the **three-stage pipeline** described in the paper:

1. **World Model Trajectory Generation**  
2. **Spatial Supervision Construction**  
3. **Post-training (SFT → GRPO)**  

---

## 📁 Repository Structure

```bash
code/
├── README.md
├── 01_world_model_trajectory_generation
│   ├── hy_worldplay
│   │   ├── hyvideo/
│   │   ├── motion_configs/
│   │   │   └── simulate_datagen.yaml
│   │   └── run_hy_worldplay_scene_generation.sh
│   ├── svc_teacher
│   │   ├── config/
│   │   │   ├── default.yaml
│   │   │   ├── simulate_camera.yaml
│   │   │   └── svc_teacher_generation.yaml
│   │   ├── datagen/
│   │   ├── run_svc_real_scene_generation.sh
│   │   └── run_svc_simulate_scene_generation.sh
│   └── tools
│       ├── build_manifest.py
│       └── export_hy_motion_plan.py
├── 02_spatial_supervision_construction
│   ├── run_build_a1_a4_motion_supervision.sh
│   ├── run_build_d1_d4_object_supervision.sh
│   ├── svc_dataset_gen/
│   └── tools
│       ├── build_motion_tasks.py
│       ├── build_object_tasks.py
│       ├── cache_detector_tracks.py
│       └── package_worldvlm_data.py
└── 03_post_training
    ├── grpo_support
    │   ├── config.yaml
    │   └── prepare_grpo_data.py
    ├── reward
    │   └── worldvlm_reward.py
    ├── run_stage1_sft.sh
    └── run_stage2_grpo.sh
```
---

### 🧪 Stage 1: World Model Trajectory Generation

This stage generates multi-step camera trajectories and corresponding scene data using world models.

🔹 Components

hy_worldplay/

* Simulated trajectory generation using HY-WorldPlay
* Includes motion configuration and runtime modules

svc_teacher/

* Teacher pipeline for both real and simulated scenes
* Handles:
    * trajectory construction
    * frame generation
    * scene serialization

tools/

* Manifest building and motion plan conversion utilities

---

### 🧠 Stage 2: Spatial Supervision Construction

Transforms generated scenes into eight task types:

* Motion-centric: A1–A4
* Object-centric: D1–D4

🔹 Key Scripts

* run_build_a1_a4_motion_supervision.sh
* run_build_d1_d4_object_supervision.sh

🔹 Modules

svc_dataset_gen/

* Prompt construction
* Bounding box normalization
* Detector integration
* Task template generation

tools/

* Task building utilities
* Data packaging for training

---

### 🔥 Stage 3: Post-training

Implements the two-stage training pipeline:

🥇 Stage 1: SFT (Supervised Fine-Tuning)
```
run_stage1_sft.sh
```
🥈 Stage 2: GRPO (Reinforcement Learning)
```
run_stage2_grpo.sh
```
🔹 Additional Components

* prepare_grpo_data.py – data preprocessing
* config.yaml – GRPO configuration
* worldvlm_reward.py – task-aware reward function

---

## 🚀 Getting Started

⚠️ This is a core release. Some engineering utilities and dependencies will be included in the full release.

1. Clone the repository
```
git clone https://github.com/your-repo/world2vlm.git
cd world2vlm
```

2. Prepare dataset

Download from:
👉 https://huggingface.co/datasets/WanyueZhang/World2VLM

3. Run pipeline

Follow the stages:

* Stage 1 → trajectory generation
* Stage 2 → supervision construction
* Stage 3 → training

---

## 📌 Notes

* ✅ This repo contains essential components for reproducing experiments
* 🔜 Full codebase (training infrastructure, optimizations, etc.) coming soon
* 🧪 Designed for research and reproducibility

---

## 📜 Citation

If you find this work useful, please cite:
```
@misc{zhang2026world2vlmdistillingworldmodel,
      title={World2VLM: Distilling World Model Imagination into VLMs for Dynamic Spatial Reasoning}, 
      author={Wanyue Zhang and Wenxiang Wu and Wang Xu and Jiaxin Luo and Helu Zhi and Yibin Huang and Shuo Ren and Zitao Liu and Jiajun Zhang},
      year={2026},
      eprint={2604.26934},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.26934}, 
}
```
---

## 🤝 Acknowledgements

We thank the open-source community and prior work on:

* World Models 🌍
* Vision-Language Models 👁️🗣️
* Embodied AI & Spatial Reasoning 🧭

---

## ⭐ Star This Repo

If you find this project helpful, consider giving it a ⭐ to support our work!

---


