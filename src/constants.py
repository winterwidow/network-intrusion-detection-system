"""dataset column definitions, attack mappings, and default file paths"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_DATA_PATH = DATA_DIR / "KDDTrain+.txt"
TEST_DATA_PATH = DATA_DIR / "KDDTest+.txt"
DEFAULT_MODEL_PATH = MODELS_DIR / "nsl_kdd_intrusion_model.joblib"
DEFAULT_ANOMALY_MODEL_PATH = MODELS_DIR / "nsl_kdd_anomaly_model.joblib"
DEFAULT_METRICS_PATH = REPORTS_DIR / "evaluation_metrics.json"
DEFAULT_ANOMALY_METRICS_PATH = REPORTS_DIR / "anomaly_evaluation_metrics.json"

RAW_LABEL_COLUMN = "label"
TARGET_COLUMN = "attack_class"
DIFFICULTY_COLUMN = "difficulty_level"

NSL_KDD_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    RAW_LABEL_COLUMN,
    DIFFICULTY_COLUMN,
]

CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]
MODEL_FEATURE_COLUMNS = [column for column in NSL_KDD_COLUMNS if column != RAW_LABEL_COLUMN]

ATTACK_CLASSES = {
    "normal": ["normal"],
    "dos": [
        "apache2",
        "back",
        "land",
        "mailbomb",
        "neptune",
        "pod",
        "processtable",
        "smurf",
        "teardrop",
        "udpstorm",
        "worm",
    ],
    "probe": ["ipsweep", "mscan", "nmap", "portsweep", "saint", "satan"],
    "r2l": [
        "ftp_write",
        "guess_passwd",
        "httptunnel",
        "imap",
        "multihop",
        "named",
        "phf",
        "sendmail",
        "snmpgetattack",
        "snmpguess",
        "spy",
        "warezclient",
        "warezmaster",
        "xlock",
        "xsnoop",
    ],
    "u2r": [
        "buffer_overflow",
        "loadmodule",
        "perl",
        "ps",
        "rootkit",
        "sqlattack",
        "xterm",
    ],
}

ATTACK_CLASS_TO_ID = {
    "normal": 0,
    "dos": 1,
    "probe": 2,
    "r2l": 3,
    "u2r": 4,
}

NORMAL_CLASS_ID = ATTACK_CLASS_TO_ID["normal"]
ATTACK_ID_TO_CLASS = {value: key for key, value in ATTACK_CLASS_TO_ID.items()}
ATTACK_TO_CLASS_ID = {
    attack: ATTACK_CLASS_TO_ID[class_name]
    for class_name, attacks in ATTACK_CLASSES.items()
    for attack in attacks
}
