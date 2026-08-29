# Network Intrusion Detection System

This project builds a Network Intrusion Detection System (IDS) using the NSL-KDD dataset. It combines:

- a multiclass intrusion classifier for known attack types
- an anomaly detector that learns from normal traffic and identifies unusual activity
- a prediction pipeline that can score new CSV data

The repository is designed for experimentation and quick evaluation of intrusion-detection models using tabular networking features.

## What is an IDS?

An Intrusion Detection System monitors network traffic and system activity to detect suspicious or unauthorized behavior. It helps identify attacks such as malware, DoS/DDoS traffic, brute-force attempts, and reconnaissance activity.

---

## Dataset

The project uses the NSL-KDD benchmark dataset stored in `data/`:

- `data/KDDTrain+.txt` - training set
- `data/KDDTest+.txt` - testing set

The dataset contains network connection records with 41 features plus the raw label column. The code maps these labels into broader classes:

- `normal`
- `dos`
- `probe`
- `r2l`
- `u2r`

The raw dataset includes fields such as:

- protocol type
- service
- flag
- byte counts and connection statistics
- connection-level counts and error rates
- final label and difficulty metadata

---

## Project structure

```text
.
├── data/
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
├── models/
│   ├── nsl_kdd_intrusion_model.joblib
│   └── nsl_kdd_anomaly_model.joblib
├── reports/
│   ├── evaluation_metrics.json
│   └── anomaly_evaluation_metrics.json
├── src/
│   ├── __init__.py
│   ├── constants.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   ├── train_anomaly.py
│   ├── evaluate.py
│   ├── evaluate_anomaly.py
│   └── predict.py
├── LICENSE
├── README.md
├── requirements.txt
└── notebooks/
    ├── eda.ipynb
    └── feature_engg.ipynb
```

### Key files

- `src/constants.py` - dataset column definitions, attack mappings, and default file paths
- `src/data_loader.py` - loads the raw NSL-KDD text files into pandas DataFrames
- `src/preprocess.py` - converts labels, identifies feature columns, and prepares preprocessing transformers
- `src/train.py` - trains the multiclass intrusion model
- `src/train_anomaly.py` - trains an Isolation Forest on normal traffic only
- `src/evaluate.py` - reports multiclass performance metrics
- `src/evaluate_anomaly.py` - reports anomaly detection metrics such as ROC-AUC
- `src/predict.py` - loads a CSV and writes predictions

---

## Data preprocessing and handling

The preprocessing pipeline is implemented in `src/preprocess.py` and is applied before training.

### 1. Loading the data

`src/data_loader.py` reads the raw `.txt` files with pandas using the NSL-KDD column names from `src/constants.py`.

### 2. Mapping attack labels

The raw label column is converted into a more manageable target using a mapping table:

- `normal` -> 0
- `dos` -> 1
- `probe` -> 2
- `r2l` -> 3
- `u2r` -> 4

This is done by mapping class names such as `neptune`, `satan`, `guess_passwd`, and `buffer_overflow` into the broader category IDs.

### 3. Feature separation

The model excludes the raw label column and the computed target from the feature set. Categorical fields are separated from numeric fields:

- categorical: `protocol_type`, `service`, `flag`
- numeric: all remaining non-categorical features

### 4. Feature encoding and scaling

The pipeline uses:

- `StandardScaler()` for numeric features
- `OneHotEncoder(handle_unknown="ignore")` for categorical features

This ensures that the model receives a consistent numeric representation for both continuous and categorical inputs.

### 5. Anomaly detection setup

For anomaly detection, only the `normal` traffic samples are used to train an `IsolationForest` model. This makes the detector learn a representation of normal behavior and flag abnormal samples as anomalies.

---

## Setup

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Required dependencies are listed in `requirements.txt`:

- pandas
- scikit-learn
- joblib

---

## Training the intrusion model

Train the default model (random forest):

```powershell
.\.venv\Scripts\python.exe -m src.train
```

This saves the pipeline to:

```text
models/nsl_kdd_intrusion_model.joblib
```

Available model choices are:

- `random_forest` (default)
- `decision_tree`
- `extra_trees`
- `logistic_regression`

Train an alternative model for testing:

```powershell
.\.venv\Scripts\python.exe -m src.train --model decision_tree
```

---

## Model evaluation

Evaluate the intrusion classifier on the test set:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate
```

This writes detailed metrics to:

```text
reports/evaluation_metrics.json
```

The evaluation includes:

- overall accuracy
- per-class precision, recall, and F1-score
- confusion matrix
- class label mapping

---

## Anomaly detection

The supervised classifier predicts known attack categories. The anomaly detector complements this by identifying unusual records that do not match normal traffic behavior.

Train the anomaly model:

```powershell
.\.venv\Scripts\python.exe -m src.train_anomaly
```

This saves the anomaly detector to:

```text
models/nsl_kdd_anomaly_model.joblib
```

Evaluate it as a normal-vs-anomaly detector:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate_anomaly
```

Metrics are written to:

```text
reports/anomaly_evaluation_metrics.json
```

The anomaly evaluation reports:

- ROC-AUC
- binary classification report for `normal` vs `anomaly`
- confusion matrix

---

## Prediction

You can run predictions on a CSV file with either:

- a headered CSV containing the model feature columns
- a raw NSL-KDD file with all columns
- a headerless feature-only file matching the model input shape

```powershell
.\.venv\Scripts\python.exe -m src.predict input.csv reports/predictions.csv
```

If the anomaly model exists, the output also includes:

- `predicted_attack_class_id`
- `predicted_attack_class`
- `is_anomaly`
- `anomaly_score`

Prediction output is saved to `reports/predictions.csv` in the example above.

---

## Output artifacts

After training and evaluation, the project generates:

- `models/nsl_kdd_intrusion_model.joblib` - trained multiclass model
- `models/nsl_kdd_anomaly_model.joblib` - trained anomaly detector
- `reports/evaluation_metrics.json` - multiclass classification metrics
- `reports/anomaly_evaluation_metrics.json` - anomaly detection metrics

