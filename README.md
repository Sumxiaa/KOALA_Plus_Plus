
# Multi-Task Optimizer Benchmark: Image Classification & Language Modeling

This repository is built on top of the [AdaFisher benchmark](https://github.com/AtlasAnalyticsLab/AdaFisher) and supports two tasks:

- **Task 1**: Image Classification  
- **Task 2**: Language Modeling

---

## 🔧 Setup Instructions

To get started, first clone the AdaFisher benchmark repository and set up the environment as described in their instructions:

```bash
git clone https://github.com/AtlasAnalyticsLab/AdaFisher.git
cd AdaFisher
# Follow their README to install the required dependencies
```

---

## 🖼 Task 1: Image Classification

Navigate to the `Task1_Image_Classification` directory. This task supports training on both CIFAR-10 and CIFAR-100 datasets.

### Run Training

- To train on **CIFAR-10**:

```bash
bash train_cifar10.sh
```

- To train on **CIFAR-100**:

```bash
bash train_cifar100.sh
```

### Notes

- You can modify the optimizer, learning rate, and other hyperparameters directly within the respective `.sh` script files.
- All optimizers from AdaFisher (e.g., AdaFisher, SGD, Adam, etc.) are supported.

---

## 🧠 Task 2: Language Modeling

Navigate to the `Task2_Language_Model` directory.

### Run Training

Simply run the corresponding training script to begin training your language model:

```bash
bash train_language_model.sh
```

> The script will use the configuration set inside to launch the training procedure, and you can modify the script for different optimizers or hyperparameter settings.

---

## 📁 Directory Structure

```
.
├── Task1_Image_Classification/
│   ├── train_cifar10.sh
│   └── train_cifar100.sh
└── Task2_Language_Model/
    └── train_language_model.sh
```
## Citation

If you make use of our work, please cite our paper:

```
@article{xia2025koala++,
  title={KOALA++: Efficient Kalman-Based Optimization of Neural Networks with Gradient-Covariance Products},
  author={Xia, Zixuan and Davtyan, Aram and Favaro, Paolo},
  journal={arXiv preprint arXiv:2506.04432},
  year={2025}
}
---

## 📌 Reminder

Make sure the environment is properly set up using the AdaFisher repository instructions before running any scripts.

Happy optimizing!
