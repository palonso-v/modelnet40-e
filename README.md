# ModelNet40-E & ScanObjectNN-E  
### Robustness and Uncertainty Benchmark for Point Cloud Classification

This repository provides:

- **ModelNet40-E** and **ScanObjectNN-E**: point cloud benchmarks with LiDAR-inspired noise
- Training and evaluation pipelines for multiple architectures:
  - PointNet
  - PointNet++
  - DGCNN
  - PointMLP
  - CurveNet
  - SimpleView
  - Point Transformer v3 (PTv3)
- Metrics for:
  - Accuracy
  - Calibration (ECE)
  - Error detection (AUROC)
  - Uncertainty awareness (correlation with ground-truth σ)

---

## 🚀 Overview

Modern point cloud benchmarks assume clean data.  
This project introduces realistic sensor noise and evaluates:

- robustness to perturbations  
- calibration under distribution shift  
- alignment between predicted uncertainty and measurement uncertainty  

We move beyond accuracy-based evaluation by explicitly modeling measurement uncertainty and its interaction with model confidence.

---

## 📂 Project Structure
.
├── train.py
├── evaluate.py
├── generate_dataset.py
├── trainer.py
├── evaluator.py
├── dataset_generator.py
├── data_utils.py
├── metrics.py
├── utils.py
├── models/


---

## 📦 Installation

```bash
conda create -n pcbench python=3.9
conda activate pcbench

pip install torch numpy matplotlib scikit-learn h5py

---

## 🧪 Dataset Generation

ModelNet40-E
python generate_dataset.py \
    --dataset modelnet40 \
    --input_root objdata/ModelNet40 \
    --output_root modelnet40-e \
    --severity all
ScanObjectNN-E
python generate_dataset.py \
    --dataset ScanObjectNN \
    --input_root h5_files/main_split \
    --output_root ScanObjectNN-e \
    --severity all

## 🏋️ Training

Example:

python train.py \
    --dataset modelnet40 \
    --model PointNet \
    --noise_level none

Available options:

--model {PointNet, PointNet2, DGCNN, PointMLP, CurveNet, SimpleView, PTv3}
--noise_level {none, light, moderate, heavy}

## 📊 Evaluation

Evaluate a trained model:

python evaluate.py \
    --dataset modelnet40 \
    --model PointNet \
    --checkpoint checkpoints/modelnet40e_none_PointNet_best.pth \
    --noise_level all

## 📈 Metrics

We evaluate:

Accuracy
ECE (Expected Calibration Error)
AUROC (error detection)
Uncertainty awareness
Pearson correlation between:
ground-truth σ
predicted uncertainty (1 − p_max)
🔬 Noise Model

We simulate LiDAR-like noise including:

range-dependent noise
angle-dependent noise
systematic bias
random outliers

Important:
The noise formulation is fixed to ensure reproducibility of the reported results.

## 🧠 Key Idea

Instead of treating noise as random corruption, we model:

measurement uncertainty at the point level

and evaluate whether models can:

remain robust
stay calibrated
understand uncertainty

## 📌 Notes

Clean data corresponds to noise_level=none
For clean data:
σ = 0
μ = 0

## 📜 Citation

https://arxiv.org/abs/2508.01269
