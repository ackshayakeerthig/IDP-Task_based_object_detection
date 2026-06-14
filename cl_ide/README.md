# Task-Aware Object Detection with Semantic Distillation & Dynamic Gating

**A Framework for Lightweight, Task-Aware Object Detection on VEGA RISC-V Processor**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Features](#features)
4. [Installation](#installation)
5. [Dataset Preparation](#dataset-preparation)
6. [Training](#training)
7. [Inference](#inference)
8. [Quantization & Deployment](#quantization--deployment)
9. [Results & Metrics](#results--metrics)
10. [References](#references)

---

## 🎯 Overview

This project implements a **task-aware object detection system** that leverages:

- **YOLOv8-Small** backbone with feature interception at P4 (20×20 grid)
- **Knowledge Distillation** from CLIP (ViT-B/32) teacher model
- **Dynamic Gating** mechanism for task-driven feature modulation
- **INT8 Quantization** for FPGA deployment on VEGA RISC-V processor
- **Semantic Affordance Learning** - maps 80 COCO classes to 14 functional tasks

### Target Metrics

| Metric | Target | Method |
|--------|--------|--------|
| **mAP (Detection)** | > 0.45 | Standard YOLOv8 evaluation |
| **Task Success Rate** | > 85% | % correct task-relevant objects detected |
| **Inference Latency** | < 20ms | Real-time on VEGA processor |
| **Power Savings** | ~30% | Via dynamic gating + sparsity |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT IMAGE (640×640)                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │   YOLOv8-Small Backbone            │
        │  ┌──────────────────────────────┐  │
        │  │ Stem (Conv 3×3)              │  │
        │  │ C2f Stage 1,2,3,4            │  │
        │  │ SPPF                         │  │
        │  └──────────────────────────────┘  │
        │           │                        │
        │           │ P4 Features (20×20)    │
        │           ▼                        │
        │  ┌──────────────────────────────┐  │
        │  │ Feature Map [B, 512, 20, 20] │  │
        │  └──────────────────────────────┘  │
        └────────────────────────────────────┘
                 ▲                ▲
                 │                │
                 │                │
        ┌────────┴────┐    ┌──────┴──────────┐
        │             │    │                 │
        ▼             ▼    ▼                 ▼
    TASK TEXT    GATING    FEATURE        YOLO HEAD
    ("Cutting")  MODULE    PROJECTION     (Detections)
        │             │    │                 │
        │             │    │                 ▼
    ┌────────────┐   │    │          ┌──────────────┐
    │ Text       │   │    │          │ Bounding     │
    │ Tokenizer  │   │    │          │ Boxes + Conf │
    │ ↓          │   │    │          │ + Classes    │
    │ TinyLinear │   │    │          └──────────────┘
    │ Mapper     │   │    │
    │ ↓          │   │    │
    │ [512-dim   │   │    │
    │ semantic]  │   │    │
    └──────┬─────┘   │    │
           │         │    │
           └────┬────┘    │
                │         │
                ▼         ▼
            HADAMARD   PROJECT
            PRODUCT    (1×1 Conv)
                │         │
                ▼         ▼
         GATED FEATURES  [B,1,14,14]
         [B,512,20,20]   STUDENT HEATMAP
                │              │
                │              ▼
                │         ┌──────────────┐
                │         │   MSE LOSS   │
                │         │   with CLIP  │
                │         │   HEATMAP    │
                │         └──────────────┘
                │
                └─────→ DOWNSTREAM DETECTION HEAD
                        (NMS, Post-processing)
                        ↓
                    FINAL DETECTIONS
```

---

## ✨ Features

### 1. **Knowledge Distillation**
- CLIP (ViT-B/32) generates semantic heatmaps: visual-textual similarity maps
- Student (YOLO) learns to produce task-aware feature maps
- MSE + Cosine + KL divergence losses for multi-objective learning

### 2. **Dynamic Gating**
- Task-conditioned feature modulation via element-wise multiplication
- Sigmoid-activated 512-dim gate vectors
- Encourages sparsity for FPGA efficiency (30-50% multiplication skipping)

### 3. **Semantic Task Mapping**
Maps 80 COCO classes to 14 functional affordances:
- **Pouring** (cup, glass, bottle, bowl)
- **Cutting** (knife, scissors, fruit, vegetables)
- **Grasping** (backpack, handbag, ball, animals)
- **Holding** (person, phone, laptop)
- **Sitting** (chair, couch, bench, bed)
- **Carrying** (person, backpack, suitcase)
- **Pushing** (car, truck, door, cart)
- **Pulling** (door, rope, bicycle)
- **Hitting** (bat, racket, ball)
- **Throwing** (ball, frisbee)
- **Opening** (door, window, box, bottle)
- **Closing** (door, window, box)
- **Balancing** (cup, glass, book, block)
- **Stacking** (block, cup, plate, book)

### 4. **Hardware Optimization**
- **INT8 Quantization-Aware Training (QAT)** for VEGA ISA
- Sparsity regularization in gating module
- Fixed-point arithmetic compatible
- BRAM footprint optimization for Genesys-2 FPGA

---

## 📦 Installation

### Requirements

```bash
# Python 3.9+
python --version

# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics>=8.0.0
pip install clip-by-openai
pip install pycocotools
pip install tensorboard
pip install tqdm numpy pillow opencv-python

# Optional: Quantization
pip install pytorch-quantization

# Optional: Kaggle notebooks
pip install kaggle
```

### Project Structure

```
task-aware-yolo/
├── task_aware_yolo.py          # Core model architecture
├── clip_teacher.py             # CLIP teacher & distillation losses
├── coco_dataloader.py          # COCO dataset with task mapping
├── train.py                    # Training pipeline with QAT
├── inference.py                # Inference engine & quantization utils
├── main.py                     # Main runner script
├── README.md                   # This file
├── requirements.txt            # Dependencies
├── checkpoints/                # Saved models
├── logs/                       # TensorBoard logs
└── outputs/                    # Inference results
```

---

## 📊 Dataset Preparation

### COCO 2017 Download

```bash
# Download COCO dataset
cd /path/to/coco
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip

# Structure should be:
# coco/
# ├── train2017/
# ├── val2017/
# └── annotations/
#     ├── instances_train2017.json
#     └── instances_val2017.json
```

### Custom Task Mapping

Edit `COCO_TO_TASK_MAPPING` in `coco_dataloader.py` to customize affordance mapping:

```python
COCO_TO_TASK_MAPPING = {
    "cup": 0,          # Pouring
    "knife": 1,        # Cutting
    "backpack": 2,     # Grasping
    # ... add more mappings
}
```

---

## 🚀 Training

### Basic Training

```python
from train import TrainingConfig, TaskAwareYOLOTrainer
from task_aware_yolo import TaskAwareYOLOWithHooks
from clip_teacher import CLIPTeacherModel
from coco_dataloader import create_coco_dataloader

# Configuration
config = TrainingConfig()
config.epochs = 100
config.batch_size = 32
config.initial_lr = 1e-3

# Load YOLO
from ultralytics import YOLO
yolo_base = YOLO('yolov8s.pt')

# Initialize models
student_model = TaskAwareYOLOWithHooks(yolo_model=yolo_base)
teacher_model = CLIPTeacherModel(model_name="ViT-B/32")

# Create dataloaders
train_loader = create_coco_dataloader(
    coco_dir='/path/to/coco',
    split='train',
    batch_size=32
)
val_loader = create_coco_dataloader(
    coco_dir='/path/to/coco',
    split='val',
    batch_size=32
)

# Train
trainer = TaskAwareYOLOTrainer(
    model=student_model,
    teacher_model=teacher_model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    device='cuda'
)
trainer.train()
```

### QAT (Last 10 Epochs)

Quantization-aware training is automatically enabled in the last 10 epochs:

```python
# In train.py, trainer.train() automatically:
# 1. Trains normally for 90 epochs (FP32)
# 2. Enables INT8 QAT for last 10 epochs
# 3. Saves best checkpoint as "model_best.pt"
```

### Kaggle TPU Training

```bash
# Create Kaggle notebook with:
import os
os.environ['KAGGLE_DATA_FOLDER'] = '/kaggle/input'
device = 'tpu:0'  # Auto-detected

# Rest of training code...
```

---

## 🔍 Inference

### Single Image Inference

```python
from inference import TaskAwareYOLOInference

# Load model
inference_engine = TaskAwareYOLOInference(
    model_path='checkpoints/model_best.pt',
    device='cuda'
)

# Run inference
import torch
from PIL import Image
import torchvision.transforms as transforms

image = Image.open('sample.jpg')
image_tensor = transforms.ToTensor()(image).unsqueeze(0)

detections = inference_engine(
    images=image_tensor,
    task_names=['cutting'],  # Task context
    conf_threshold=0.5
)

# Print results
for det in detections:
    print(f"Box: {det['bbox']}, Conf: {det['confidence']:.2f}")
```

### Batch Inference

```python
# Batch processing
images_batch = torch.stack([
    transforms.ToTensor()(Image.open(f'image_{i}.jpg'))
    for i in range(4)
])

tasks = ['pouring', 'cutting', 'grasping', 'holding']

detections_batch = inference_engine(
    images=images_batch,
    task_names=tasks,
    conf_threshold=0.5
)

# Get performance stats
stats = inference_engine.get_inference_stats()
print(f"Mean latency: {stats['mean_latency_ms']:.2f} ms")
print(f"Sparsity: {stats['mean_sparsity']:.1%}")
```

---

## 🔧 Quantization & Deployment

### INT8 Quantization

```python
from inference import QuantizationCalibrator, QuantizationConfig

# Setup quantization
config = QuantizationConfig()
calibrator = QuantizationCalibrator(config)

# Calibrate on small dataset
calibrator.calibrate(model, calibration_loader)

# Get INT8 model
int8_model = Int8Model(model, scales=calibrator.scales)
```

### Export to ONNX (for FPGA toolchain)

```python
inference_engine.to_onnx('model_int8.onnx')

# Use ONNX in Xilinx Vivado HLS or Vitis
# Custom C/RTL wrapper for VEGA integration
```

### FPGA Resource Estimation

```python
from inference import estimate_fpga_resources

resources = estimate_fpga_resources(model)
print(f"BRAM usage: {resources['estimated_bram_kb']:.0f} KB")
print(f"Fits in Genesys-2: {resources['can_fit_in_bram']}")

# Genesys-2 resources:
# - 19 Mb BRAM (2375 KB)
# - 200k LUTs
# - 440 DSPs
```

---

## 📈 Results & Metrics

### Training Logs (TensorBoard)

```bash
tensorboard --logdir=logs/
# Open http://localhost:6006
```

Monitor:
- **Loss curves** (YOLO, Distillation, Total)
- **Learning rate schedule**
- **Per-component losses** (MSE, Cosine, KL)
- **Sparsity evolution**

### Evaluation Metrics

```python
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Evaluate on COCO val2017
coco_gt = COCO('/path/to/annotations/instances_val2017.json')
coco_dt = coco_gt.loadRes('predictions.json')

coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

print(f"mAP@.5:.95: {coco_eval.stats[0]:.4f}")
print(f"mAP@.5: {coco_eval.stats[1]:.4f}")
print(f"mAP@.75: {coco_eval.stats[2]:.4f}")
```

### Task Success Rate

```python
# Compute % of images where task-relevant object is top detection
task_success_count = 0
total_images = 0

for image_id, task_name in task_image_pairs:
    detections = inference_engine([image_id], task_names=[task_name])
    top_det = detections[0][0]  # First detection
    
    # Check if class matches task
    if is_relevant_to_task(top_det['class_id'], task_name):
        task_success_count += 1
    
    total_images += 1

task_success_rate = task_success_count / total_images * 100
print(f"Task Success Rate: {task_success_rate:.1f}%")
```

---

## 🔬 Ablation Studies

### Study 1: Gating vs No-Gating

```python
# Train TaskAwareYOLO with/without gating module
config_with_gating = TrainingConfig()
config_without_gating = TrainingConfig()
config_without_gating.disable_gating = True

# Compare metrics
```

### Study 2: Distillation Weights

```python
# Vary lambda in L_total = L_YOLO + λ * L_Distill
lambdas = [0.1, 0.3, 0.5, 0.7, 1.0]

for lam in lambdas:
    config = TrainingConfig()
    config.distill_weight = lam
    # Train and evaluate
```

### Study 3: Quantization Impact

```python
# Compare FP32 vs INT8 metrics
fp32_model = load_model('checkpoints/model_best_fp32.pt')
int8_model = load_model('checkpoints/model_best_int8.pt')

# Evaluate both on COCO val
```

---

## 📚 References

1. **Task-Aware Object Detection**:
   - "What Object Should I Use?" (arXiv:1904.03000) - Functional task taxonomy
   - YOLOv8: https://github.com/ultralytics/ultralytics

2. **Knowledge Distillation**:
   - Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
   - FitNet, Attention Transfer, etc.

3. **CLIP & Vision-Language**:
   - Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (2021)

4. **Quantization**:
   - Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (2018)
   - QAT guidelines for INT8

5. **VEGA & FPGA**:
   - RISC-V ISA Specification
   - Genesys-2 FPGA Board Documentation
   - Xilinx Vivado HLS

---

## 🤝 Contributing

This is an academic research project. For questions/improvements:

1. **Code Issues**: Document clearly with examples
2. **New Features**: Submit with test cases
3. **Performance Tips**: Share benchmark results

---

## 📝 Citation

If you use this framework, please cite:

```bibtex
@misc{task_aware_yolo_2024,
  title={Task-Aware Object Detection with Semantic Distillation and Dynamic Gating for VEGA RISC-V},
  author={Your Name},
  institution={RV College of Engineering},
  year={2024}
}
```

---

## 📞 Contact

**Department of Computer Science & Engineering**
RV College of Engineering, Bangalore

**Faculty Advisor**: [Advisor Name]
**Student Team**: [Names]

---

## ⚖️ License

This project is released under the [MIT License](LICENSE).

---

**Last Updated**: 2024
**Status**: Active Development
