# DHDA: Accurately Modeling Configuration Performance under Dynamic Drifts

This repository contains the code, datasets, and raw experimental results for the paper:

> **DHDA: Accurately Modeling Configuration Performance under Dynamic Drifts**

## Introduction

Modern configurable software systems rely on performance models to relate configurations to system performance. However, changing workloads, hardware, and software can cause dynamic global and local concept drifts, which existing offline and transfer learning approaches cannot reliably handle because they assume static models or explicit environment transitions. To address this challenge, we propose DHDA, an online configuration performance learning framework that adapts to concept drifts through dually hierarchical adaptation. At the upper level, DHDA dynamically redivides the configuration space and retrains local models only when global drifts are detected. At the lower level, each local model asynchronously detects and adapts to local drifts within its own region. To balance responsiveness and efficiency, DHDA combines incremental updates with periodic retraining to minimize unnecessary computation during stable periods. Furthermore, DHDA uses a consistency-guided preservation-and-reuse mechanism for recurring or data-scarce drifts, enabling reliable historical knowledge reuse while avoiding inconsistent concept transfer. We evaluate DHDA on twelve widely used configurable software systems against state-of-the-art approaches. DHDA achieves the best median accuracy on 7/12 systems and the best statistical rank on 9/12, reducing prediction error by up to 2.5x with modest adaptation overhead. Concept preservation improves robustness under recurring drifts, and the framework supports different local learners.

## Repository Structure

```text
.
├── DaLModels.py              # Core DHDA implementation
├── base_model/
│   └── basemodels.py         # Local and online base learners
├── configs.py                # Model configuration
├── dataset/                  # Datasets for the 12 subject systems
│   ├── data1 ... data8       # Original systems
│   └── data9 ... data12      # Additional systems
├── RQs/
│   ├── RQ1/                  # State-of-the-art comparison
│   ├── RQ2/                  # Component-ablation scripts
│   ├── RQ3/                  # Local-model scripts
│   └── RQ4/                  # Parameter-sensitivity scripts
├── result/
│   ├── rq1 ... rq5           # Raw and processed experimental results
│   └── scripts/              # Result-processing scripts
├── test/
│   └── test_similarity.py    # Similarity computation used for model reuse
└── requirements.txt          # Core dependency versions
```

Each RQ directory contains two scripts:

- `test_rq*_original.py` runs the experiment on the eight original systems (`data1`-`data8`).
- `test_rq*_new.py` runs the experiment on the four additional systems (`data9`-`data12`).

## Subject Systems and Datasets

| Dataset | Subject system | Dataset group |
| --- | --- | --- |
| `data1` | SaC | Original |
| `data2` | x264 | Original |
| `data3` | Storm | Original |
| `data4` | SPEAR | Original |
| `data5` | SQLite | Original |
| `data6` | Nginx | Original |
| `data7` | ExaStencils | Original |
| `data8` | DeepArch | Original |
| `data9` | ImageMagick | Additional |
| `data10` | libvips | Additional |
| `data11` | Pulsar | Additional |
| `data12` | RabbitMQ | Additional |

All dataset paths in the experiment scripts are resolved relative to the repository root. Place the extracted datasets under `dataset/data1` through `dataset/data12` before running the experiments.

## Prerequisites and Installation

The experiments were prepared for Python 3.9. We recommend creating an isolated environment:

```bash
conda create -n dhda-ext python=3.9
conda activate dhda-ext
```

Install the required packages and their specified versions:

```bash
pip install \
  imbalanced-learn==0.10.1 \
  numpy==1.22.4 \
  pandas==2.2.3 \
  river==0.21.2 \
  rpy2==3.5.11 \
  scikit-learn==1.2.2 \
  scikit-multiflow==0.5.3 \
  matplotlib scipy ruptures
```

Clone the repository and enter its root directory:

```bash
git clone https://github.com/ideas-labo/dhda-ext.git
cd dhda-ext
```

## Running the Experiments

Run scripts as Python modules from the repository root. For example, to run RQ1:

```bash
python -m RQs.RQ1.test_rq1_original
python -m RQs.RQ1.test_rq1_new
```

The scripts automatically create the required `log`, `csv`, and `figures` directories below their configured result directory.

The experiments repeat each setting 30 times and can therefore take considerable time. The random seed, number of repetitions, tested learners, and DHDA parameters can be changed near the main experiment loop in each script.

## Research-Question Reproduction

### RQ1: Comparison with state-of-the-art approaches

RQ1 evaluates prediction accuracy and adaptation cost against state-of-the-art and representative online approaches.

```bash
python -m RQs.RQ1.test_rq1_original
python -m RQs.RQ1.test_rq1_new
```

Published RQ1 accuracy, runtime, and statistical summaries are available under [`result/rq1`](result/rq1/). The released result files also contain the externally obtained SeMPL, BEETLE, and SELeCT measurements used in the paper comparison.

### RQ2: DHDA with different local models

RQ2 studies DHDA with RF, HT, kNN, and LR local learners and compares each DHDA-enhanced learner with its standalone counterpart.

The corresponding experiment implementation is currently stored in the local-model scripts:

```bash
python -m RQs.RQ3.test_rq3_original
python -m RQs.RQ3.test_rq3_new
```

Published results are available under [`result/rq2`](result/rq2/).

### RQ3: Contributions of the key DHDA components

RQ3 evaluates component-ablation variants, including variants without upper-level adaptation, lower-level adaptation, hybrid updating, and concept-transfer functionality.

The corresponding experiment implementation is currently stored in the ablation scripts:

```bash
python -m RQs.RQ2.test_rq2_original
python -m RQs.RQ2.test_rq2_new
```

Published results are available under [`result/rq3`](result/rq3/).

### RQ4: Parameter sensitivity

RQ4 studies the sensitivity of DHDA to its alignment-frequency parameter, denoted by alpha in the paper.

```bash
python -m RQs.RQ4.test_rq4_original
python -m RQs.RQ4.test_rq4_new
```

Accuracy and runtime results are available under [`result/rq4`](result/rq4/).

### RQ5: Model adaptation time

RQ5 compares model-adaptation time normalized by the number of timesteps in each stream. The processed values are available under [`result/rq5/normalized_runtime`](result/rq5/normalized_runtime/), while the corresponding raw total runtimes are stored under [`result/rq1/runtime`](result/rq1/runtime/).

## Compared Approaches and Papers

The paper compares DHDA with transfer-learning, meta-learning, and concept-drift adaptation methods. The following are the papers for the three externally evaluated approaches:

- **SeMPL** — Jingzhi Gong and Tao Chen, *Predicting Configuration Performance in Multiple Environments with Sequential Meta-Learning*, FSE 2024. [Paper](https://doi.org/10.1145/3643743) | [Repository](https://github.com/ideas-labo/SeMPL)
- **BEETLE** — Rahul Krishna, Vivek Nair, Pooyan Jamshidi, and Tim Menzies, *Whence to Learn? Transferring Knowledge in Configurable Systems Using BEETLE*, IEEE Transactions on Software Engineering. [Paper](https://doi.org/10.1109/TSE.2020.2983927) | [Repository](https://github.com/ai-se/BEETLE)
- **SELeCT** — Ben Halstead, Yun Sing Koh, Patricia Riddle, Mykola Pechenizkiy, and Albert Bifet, *A Probabilistic Framework for Adapting to Changing and Recurring Concepts in Data Streams*, IEEE DSAA 2022. [Paper](https://doi.org/10.1109/DSAA54385.2022.10032368) | [Open-access preprint](https://arxiv.org/abs/2408.09324)

The comparison also includes Adaptive Random Forest (ARF), Streaming Random Patches (SRP), the original ICSE version of DHDA, and DaL.

## Experimental Results

The `result` directory contains the publication-facing raw and processed data for all five research questions. Each system-level CSV contains results from 30 experimental runs. See [`result/README.md`](result/README.md) and the README inside each RQ result directory for column mappings and reproduction details.

To regenerate the RQ1 summary table, run:

```bash
python result/scripts/build_rq1_stats.py
```

