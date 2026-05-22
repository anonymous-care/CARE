# Inactive-User-Retention(CARE)

This repository contains the core code for our offline experiments on retention-oriented recommendation for at-risk users in social platforms.

The goal of this project is not to propose a new recommendation backbone, but to study whether a recommendation intervention can improve the visibility and connection opportunities of inactive / at-risk users while keeping overall utility drift controlled.

## 1. Overview

The pipeline has two stages:

1. **Dataset preprocessing**
   - build the graph
   - compute user activity statistics
   - identify inactive / at-risk users
   - generate processed inputs for the main experiments

2. **Main experiment pipeline**
   - train backbone models
   - run reweighted variants
   - test trained models
   - run reranking baselines
   - run the proposed reranking method

This repository reproduces an **offline experimental setting** only. It does **not** include live deployment or online A/B testing.

---

## 2. Environment Setup

We recommend Python 3.9 or 3.10.

Install dependencies with:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

Or with conda:

```bash
conda create -n inactive-retention python=3.10
conda activate inactive-retention
pip install -r requirements.txt
```

---

## 3. Dataset Preparation

### 3.1 Download the dataset

Download the dataset from:

```text
https://zenodo.org/records/19487502
```

Extract the downloaded files under the repository's `dataset/` directory.

### 3.2 Configure paths

Some scripts use manually specified path variables such as:

```python
currDir = ''
outputDir = ''
```

Before running the code, update these variables so that they correctly point to your local dataset directory and output directory.

Make sure the path configuration is consistent across all scripts.

---

## 4. Data Preprocessing

Before running the main experiments, preprocess the dataset in the following order:

### Step 1: Generate the graph

```bash
cd dataset
python generateEdges.py
```

### Step 2: Compute user activity / identify inactive users

```bash
python findInactive.py
```

### Step 3: Run analysis scripts

```bash
python figureInactivebySlope.py
python figurePopularity.py
```

Notes:

* `figureInactivebySlope.py` corresponds to the **trend-based detector** used in this project.
* `figurePopularity.py` computes the related popularity analysis used in preprocessing.

### Step 4: Generate final processed outputs

```bash
python generateRatio.py
cd ..
```

After these steps, the preprocessing stage is complete.

---

## 5. Main Experiment Pipeline

The main experiment code is organized as follows.

### Backbone training without reweighting

* `main.py` → LightGCN without reweighting
* `main_mf.py` → MF without reweighting
* `main_neumf.py` → NeuMF without reweighting

### Backbone training with reweighting

* `main_reweight.py`
* `main_reweight5.py` → LightGCN with reweighting
* `main_mf_reweight5.py` → MF with reweighting
* `main_neumf_reweight5.py` → NeuMF with reweighting

### Testing

* `test_model.py` → LightGCN
* `test_MF.py` → MF
* `test_NeuMF.py` → NeuMF

### Reranking

* `degree_debias_rerank.py` → reranking baseline
* `rerank_parrel.py` → proposed reranking method

---

## 6. How to Use `example.sh`

`example.sh` is **not** a one-click script for the full pipeline.
Instead, it contains **example commands** for running the main experiments.

You should:

1. finish the dataset preprocessing first
2. open `example.sh`
3. manually select the commands you want to run
4. execute them one by one as needed

The command examples in `example.sh` cover:

* LightGCN training / testing
* MF training / testing
* NeuMF training / testing
* reranking baseline
* proposed reranking

---

## 7. Recommended Workflow

For a fresh reproduction, we recommend the following order:

1. preprocess the dataset:

   * `generateEdges.py`
   * `findInactive.py`
   * `figureInactivebySlope.py`
   * `figurePopularity.py`
   * `generateRatio.py`

2. manually run the relevant commands from `example.sh`:

   * vanilla backbone training
   * reweighted training
   * testing
   * baseline reranking
   * proposed reranking

