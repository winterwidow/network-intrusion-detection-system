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

### 1. Mapping attack labels

The raw label column is converted into a more manageable target using a mapping table:

- `normal` -> 0
- `dos` -> 1
- `probe` -> 2
- `r2l` -> 3
- `u2r` -> 4

This is done by mapping class names such as `neptune`, `satan`, `guess_passwd`, and `buffer_overflow` into the broader category IDs.

### 2. Feature separation

The model excludes the raw label column and the computed target from the feature set. Categorical fields are separated from numeric fields:

- categorical: `protocol_type`, `service`, `flag`
- numeric: all remaining non-categorical features

### 3. Feature encoding and scaling

The pipeline uses:

- `StandardScaler()` for numeric features
- `OneHotEncoder(handle_unknown="ignore")` for categorical features

This ensures that the model receives a consistent numeric representation for both continuous and categorical inputs.

### 4. Anomaly detection setup

For anomaly detection, only the `normal` traffic samples are used to train an `IsolationForest` model. This makes the detector learn a representation of normal behavior and flag abnormal samples as anomalies.

---

## Requirements

The project uses Python and the dependencies listed in `requirements.txt`:

- pandas
- scikit-learn
- joblib

---

## Methodology

The project uses two complementary machine-learning approaches because intrusion detection has two related goals: classify attacks that are already represented in the training data and identify traffic that differs from normal behavior.

### Multiclass intrusion classification

The primary model is a Random Forest classifier. Each tree learns a set of decision rules from a sampled view of the training data, and the final prediction is selected by combining the trees' predictions.

The training process is:

1. Load the NSL-KDD training records.
2. Convert individual attack names into five broader classes: `normal`, `dos`, `probe`, `r2l`, and `u2r`.
3. Remove the raw label and use the 41 connection features plus `difficulty_level` as input features.
4. One-hot encode `protocol_type`, `service`, and `flag` so categorical values can be used by the estimator.
5. Standardize numeric features with `StandardScaler`.
6. Fit the preprocessing and Random Forest steps together in a scikit-learn pipeline.
7. Evaluate the saved pipeline on the separate NSL-KDD test set.

Random Forest was selected as the default because it handles mixed tabular features well, can model nonlinear relationships, and is less sensitive to feature scaling than many linear models. Class balancing is enabled with `balanced_subsample` to reduce the effect of uneven class frequencies.

The repository also supports Decision Tree, Extra Trees, and Logistic Regression models for comparison.

### Anomaly detection

The anomaly model is an Isolation Forest. Unlike the classifier, it is trained only on records labelled `normal`; attack labels are not used as target classes during fitting.

Isolation Forest works by repeatedly partitioning the feature space. Records that are isolated in fewer partitions are considered more unusual and receive a higher anomaly score. The model uses a contamination setting of `0.02`, representing the expected fraction of unusual records in the normal-only training data.

This approach is useful because an anomaly detector can flag behavior that does not belong to one of the known attack categories. Its output is interpreted as:

- `is_anomaly = false`: the record resembles the learned normal traffic
- `is_anomaly = true`: the record is sufficiently unusual according to the detector
- `anomaly_score`: higher values indicate more unusual behavior

The anomaly evaluation converts the test set into a binary problem: `normal` versus `anomaly`, where every known attack category is treated as an anomaly.

---

## Evaluation metrics

The results below are from the evaluation recorded in `evaluation_metrics.txt` using 22,544 test records. Precision measures how often a predicted class is correct, recall measures how many records of that class were found, and F1-score combines both measures.

### Multiclass classifier (`src.evaluate`)

| Class                |  Precision |     Recall |   F1-score |    Support |
| -------------------- | ---------: | ---------: | ---------: | ---------: |
| normal               |     0.6375 |     0.9734 |     0.7704 |      9,711 |
| dos                  |     0.9592 |     0.7559 |     0.8455 |      7,460 |
| probe                |     0.8350 |     0.6167 |     0.7094 |      2,421 |
| r2l                  |     0.8824 |     0.0104 |     0.0206 |      2,885 |
| u2r                  |     0.8000 |     0.1791 |     0.2927 |         67 |
| **Overall accuracy** |            |            | **0.7375** | **22,544** |
| **Macro average**    | **0.8228** | **0.5071** | **0.5277** | **22,544** |
| **Weighted average** | **0.7970** | **0.7375** | **0.6913** | **22,544** |

The classifier identifies normal traffic particularly well, with 97.34% recall, and performs best on DoS attacks by F1-score. The very low recall for `r2l` and `u2r` shows that the minority attack classes remain difficult to detect. This is an important limitation: overall accuracy is influenced by the larger classes and does not fully represent performance on rare attacks.

### Anomaly detector (`src.evaluate_anomaly`)

| Class                |  Precision |     Recall |   F1-score |    Support |
| -------------------- | ---------: | ---------: | ---------: | ---------: |
| normal               |     0.6686 |     0.9762 |     0.7936 |      9,711 |
| anomaly              |     0.9724 |     0.6338 |     0.7674 |     12,833 |
| **Accuracy**         |            |            | **0.7813** | **22,544** |
| **Macro average**    | **0.8205** | **0.8050** | **0.7805** | **22,544** |
| **Weighted average** | **0.8415** | **0.7813** | **0.7787** | **22,544** |

The anomaly detector achieved a ROC-AUC of **0.9510**, indicating strong separation between normal and attack records across different score thresholds. It has high anomaly precision, meaning flagged records are usually anomalous, while its 63.38% anomaly recall shows that some attacks still resemble normal traffic and are missed.

---

## Prediction

The prediction pipeline accepts:

- a headered CSV containing the model feature columns
- a raw NSL-KDD file with all columns
- a headerless feature-only file matching the model input shape

If the anomaly model exists, the output also includes:

- `predicted_attack_class_id`
- `predicted_attack_class`
- `is_anomaly`
- `anomaly_score`

Prediction output contains the original features together with the predicted class ID and class name. When available, it also contains the anomaly flag and score.

---

## Output artifacts

After training and evaluation, the project generates:

- `models/nsl_kdd_intrusion_model.joblib` - trained multiclass model
- `models/nsl_kdd_anomaly_model.joblib` - trained anomaly detector
- `reports/evaluation_metrics.json` - multiclass classification metrics
- `reports/anomaly_evaluation_metrics.json` - anomaly detection metrics
