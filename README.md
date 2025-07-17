# HuSCF-GAN: Distributed Conditional GAN with Heterogeneous U-shape Split Clustered Federated Learning

## 📌 Abstract
<!-- TODO: Add your research abstract here -->

---

## 🧠 Overview

**HuSCF-GAN** introduces a **distributed Conditional GAN (cGAN)** training framework leveraging a novel **Heterogeneous U-shape Split Clustered Federated Learning** approach. It is designed to address the following challenges in distributed machine learning:

- **Data heterogeneity**
- **Device heterogeneity**
- **Resource-constrained environments**
- **Multi-domain datasets**
- **Data sharing constraints**

This framework distributes the model across clients and a central server. The split point (cut) in the model is optimally selected using a **Genetic Algorithm**, based on each client’s device profile (e.g., CPU frequency, FLOPs per cycle, data rate).



---



### ▶️ Run Training

To start the training process:

`python train.exe`

> **Note:** Run this command from inside the project directory.

---

## ⚙️ Configurations

Training scenarios are defined in the `configs.yaml` via YAML configuration files. You can adjust:

- Dataset combinations  
- IID/Non-IID settings  
- Number of clients/domains  
- Training settings
- Training Hyperparameters

### Available Scenarios:

| Scenario Type   | Domains                          | Distribution Type         |
|-----------------|----------------------------------|---------------------------|
| Single Domain   | MNIST                            | IID / Non-IID             |
| Double Domain   | MNIST + FMNIST                   | IID / Non-IID / Highly Non-IID |
| Multi-Domain    | MNIST, FMNIST, KMNIST, NotMNIST  | IID                       |

---

## 🧩 Model Splitting & Device Profiles

HuSCF-GAN splits the generator and discriminator across clients and the server. The optimal cut is determined by a **Genetic Algorithm**, taking into account the performance characteristics of each device.

### Device Profiles

Device profiles are defined in:

`Cut_Selection/profiles.yaml`

Each profile includes:

- `cpu_frequency`: CPU clock speed (e.g., in GHz)
- `flops_per_cycle`: Computational power
- `data_rate`: Communication bandwidth

You can **edit**, **add**, or **remove** profiles to simulate various deployment environments.

---

## 📊 Evaluation Metrics

All evaluation logic is located in the `Metrics/` directory. The framework evaluates:

- **Diversity & Coverage:**  
  Inception-like score using dataset-specific classifiers to assess the variety and realism of generated images.
  
- **Classification Accuracy:**  
  Measures how well generated images preserve class-specific information.

- **Latency:**  
  Captures end-to-end system delay including computation and communication overhead.

---

## 📚 Supported Datasets

This project uses standard image classification datasets:

- MNIST
- Fashion-MNIST (FMNIST)
- Kuzushiji-MNIST (KMNIST)
- NotMNIST

These datasets are used to simulate single-domain, dual-domain, and multi-domain federated scenarios.

---
<!-- 
## 📝 Citation

If you use HuSCF-GAN or build upon it in your research, please consider citing the associated publication.

```bibtex
Add your BibTeX entry here -->
