# FITMM (ACM MM 2025)
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
| **Baby** | 原论文 (FITMM) | 0.0716 | 0.1089 | 0.0387 | 0.0484 |
| | 简化代码 (Test) | 0.0704 | 0.1078 | 0.0372 | 0.0468 |
| **Sports** | 原论文 (FITMM) | 0.0809 | 0.1187 | 0.0441 | 0.0538 |
| | 简化代码 (Test) | 0.0798 | 0.1187 | 0.0435 | 0.0535 |
| **Clothing** | 原论文 (FITMM) | 0.0698 | 0.1017 | 0.0378 | 0.0457 |
| | 简化代码 (Test) | 0.0681 | 0.0990 | 0.0372 | 0.0450 |

