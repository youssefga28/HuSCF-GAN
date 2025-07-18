# 📁 Results Directory

This folder contains all evaluation outputs generated from different training **scenarios** in the HuSCF-GAN framework. Each subfolder corresponds to a specific scenario configuration (e.g., dataset combinations and data distribution type), and includes performance metrics, plots, and latency measurements.

---

## 📂 Directory Structure

Each experiment scenario creates its own subdirectory:

results/
├── scenario_1_single_domain_IID/
│ ├── plots/
│ ├── tables/
│ └── ...
├── scenario_2_double_domain_NonIID/
│ ├── plots/
│ ├── tables/
│ └── ...
└── ...

---

## 📊 Evaluation Outputs

Each scenario folder—while empty now— After training it will include:

### 📈 `plots/`

This folder contains visualizations of performance metrics over training rounds. These include:

| File Name               | Description                                          |
|-------------------------|------------------------------------------------------|
| `mnist_scores.png`      | Generation score (inception-like) for MNIST         |
| `fmnist_scores.png`     | Generation score for Fashion-MNIST                  |
| `kmnist_scores.png`     | Generation score for KMNIST                         |
| `notmnist_scores.png`   | Generation score for NotMNIST                       |


These plots reflect the **quality and diversity** of the generated images across training rounds.

---

### 📄 `tables/`

This folder contains structured numerical results:

| File Name              | Description                                                              |
|------------------------|--------------------------------------------------------------------------|
| `final_metrics.csv`    | Classification performance on real test sets using classifiers trained on synthetic data |
| `latency.csv`          | Total system latency (in seconds) based on the selected model split      |

#### 🧪 Classification Metrics

- A classifier is trained **only on 30,000 synthetic samples per domain**.
- It is evaluated on the **real test dataset**.
- Metrics reported:
  - Accuracy
  - Precision (macro average)
  - Recall (macro average)
  - F1-score (macro average)
  - False Positive Rate (FPR macro)

These metrics measure how well synthetic data captures task-relevant features from the original data distribution.

#### ⏱ Latency

- Latency is computed as part of the **model splitting optimization**.
- It includes:
  - **Computation latency** (based on FLOPs and CPU frequency)
  - **Transmission latency** (based on output sizes and bandwidth)

Latency is calculated by the **Genetic Algorithm** implemented in `Cut_Selection/CutLayerSelection.py`.

---

## 📝 Notes

- All results are saved automatically after training.
- Metrics and classifiers are handled in `Metrics/evaluation.py`.
  - Pretrained dataset-specific classifiers are used to compute image generation scores.
  - CNNs trained on synthetic data are used to compute classification metrics.
- Latency is calculated using the device profiles in `Cut_Selection/profiles.yaml`.

---


## 🔁 Summary

| Component      | Description                                      |
|----------------|--------------------------------------------------|
| `plots/`       | Visual insights into image quality and trends    |
| `tables/`      | Final evaluation scores and latency per scenario |
| `latency.csv`  | Per-iteration end-to-end delay for model execution |

Each scenario folder provides a self-contained summary of model performance under specific settings.
