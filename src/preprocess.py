import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .constants import (
    ATTACK_ID_TO_CLASS,
    ATTACK_TO_CLASS_ID,
    CATEGORICAL_FEATURES,
    RAW_LABEL_COLUMN,
    TARGET_COLUMN,
)


def add_attack_class(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TARGET_COLUMN] = df[RAW_LABEL_COLUMN].map(ATTACK_TO_CLASS_ID)

    unknown_attacks = sorted(df.loc[df[TARGET_COLUMN].isna(), RAW_LABEL_COLUMN].unique())
    if unknown_attacks:
        raise ValueError(f"Unknown attack labels found: {unknown_attacks}")

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {RAW_LABEL_COLUMN, TARGET_COLUMN}
    return [column for column in df.columns if column not in excluded]


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    prepared = add_attack_class(df)
    return prepared[get_feature_columns(prepared)], prepared[TARGET_COLUMN]


def get_feature_types(feature_columns: list[str]) -> tuple[list[str], list[str]]:
    categorical = [column for column in CATEGORICAL_FEATURES if column in feature_columns]
    numeric = [column for column in feature_columns if column not in categorical]
    return numeric, categorical


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    numeric_features, categorical_features = get_feature_types(feature_columns)

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )


def decode_attack_class(class_id: int) -> str:
    return ATTACK_ID_TO_CLASS[int(class_id)]
