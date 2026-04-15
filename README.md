# FITMM: Adaptive Frequency-Aware Multimodal Recommendation via Information-Theoretic Representation Learning (ACM MM 2025)
---

## Installation

We provide two options to set up the conda environment.

### Option A: Install from `environment.yaml` (recommended)

```bash
conda env create -f environment.yaml
conda activate mmrec
```

### Option B: Install from `environment.txt` (explicit spec)

```bash
conda create -n mmrec --file environment.txt
conda activate mmrec
```

> If your environment name is different, replace `mmrec` with your own env name.

---

## Data

Our datasets follow the same format as **MMRec**.  
Please download the datasets and place them under the `data/` directory, then you can run experiments directly.

```text
FITMM/
├── data/
│   ├── dataset_name_1/
│   ├── dataset_name_2/
│   └── dataset_name_3/
├── src/
└── run_exp.sh
```

---

## Running

Run the following script to start training and evaluation:

```bash
sh run_exp.sh
```

To specify a GPU, pass the GPU id to the script:

```bash
sh run_exp.sh 1
```

You can also run the entrypoint directly:

```bash
cd src
python main.py -m FITMM -d baby --gpu_id 1
```

### 📊 原论文 vs. 简化版代码 (核心指标对比)

| 数据集 | 模型来源 | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baby** | 原论文 (FITMM) | 0.0752 | 0.1109 | 0.0413 | 0.0503 |
| | 简化代码 (Valid) | 0.0680 | 0.1039 | 0.0367 | 0.0458 |
| | 简化代码 (Test) | 0.0704 | 0.1078 | 0.0372 | 0.0468 |
| **Sports** | 原论文 (FITMM) | 0.0820 | 0.1218 | 0.0435 | 0.0535 |
| | 简化代码 (Valid) | 0.0796 | 0.1168 | 0.0431 | 0.0525 |
| | 简化代码 (Test) | 0.0798 | 0.1187 | 0.0435 | 0.0535 |
| **Clothing** | 原论文 (FITMM) | 0.0691 | 0.1003 | 0.0381 | 0.0459 |
| | 简化代码 (Valid) | 0.0669 | 0.1006 | 0.0362 | 0.0447 |
| | 简化代码 (Test) | 0.0681 | 0.0990 | 0.0372 | 0.0450 |