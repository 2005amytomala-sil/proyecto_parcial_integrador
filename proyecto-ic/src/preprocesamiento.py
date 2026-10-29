"""Preprocesamiento de UNSW-NB15 (Fase II, Persona A).

Todas las transformaciones se ajustan SOLO con entrenamiento: viven dentro de un
Pipeline/ColumnTransformer de scikit-learn y se aplican al conjunto de prueba con
`transform`, nunca con `fit`. Requiere scikit-learn >= 1.2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SEED = 42
TARGET = "label"                                  # 0 = normal, 1 = ataque
CAT_COLS = ["proto", "service", "state"]
# Columnas que NUNCA son predictores: identificador y variables que revelan la respuesta.
NO_PREDICTORAS = ["id", "attack_cat", "label"]


def cargar_datos(ruta_train: str, ruta_test: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee los CSV oficiales (particion de entrenamiento y de prueba)."""
    return pd.read_csv(ruta_train), pd.read_csv(ruta_test)


def separar_xy(df: pd.DataFrame):
    """Devuelve X (predictores), y (label) y attack_cat (solo para analizar errores)."""
    X = df.drop(columns=[c for c in NO_PREDICTORAS if c in df.columns])
    y = df[TARGET].astype(int)
    cat_ataque = df["attack_cat"] if "attack_cat" in df.columns else None
    return X, y, cat_ataque


def columnas_por_tipo(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Separa columnas numericas y categoricas (proto, service, state)."""
    cat = [c for c in CAT_COLS if c in X.columns]
    num = [c for c in X.columns if c not in cat]
    return num, cat


def detectar_columnas_log(X_train: pd.DataFrame, num_cols: list[str],
                          umbral_asimetria: float = 2.0) -> list[str]:
    """Columnas no negativas, no binarias y muy asimetricas (se calcula SOLO con entrenamiento)."""
    elegidas = []
    for c in num_cols:
        s = X_train[c].dropna()
        if s.nunique() > 2 and s.min() >= 0 and abs(s.skew()) > umbral_asimetria:
            elegidas.append(c)
    return elegidas


def construir_preprocesador(num_cols: list[str], cat_cols: list[str],
                            log_cols: list[str] | tuple = ()) -> ColumnTransformer:
    """Imputacion + (log1p opcional) + escalado para numericas; imputacion + one-hot para categoricas."""
    log_cols = list(log_cols)
    num_normales = [c for c in num_cols if c not in log_cols]
    bloques = []
    if num_normales:
        bloques.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("esc", StandardScaler()),
        ]), num_normales))
    if log_cols:
        bloques.append(("num_log", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("esc", StandardScaler()),
        ]), log_cols))
    if cat_cols:
        bloques.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="desconocido")),
            # categorias raras (<50 registros en entrenamiento) o nuevas en prueba se agrupan
            ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=50, sparse_output=False)),
        ]), cat_cols))
    return ColumnTransformer(bloques, remainder="drop", verbose_feature_names_out=False)


def submuestra_estratificada(df: pd.DataFrame, n: int, col_estrato: str = "attack_cat",
                             semilla: int = SEED) -> pd.DataFrame:
    """Submuestra de entrenamiento que conserva la proporcion de cada categoria de ataque."""
    if n >= len(df):
        return df.reset_index(drop=True)
    muestra, _ = train_test_split(df, train_size=n, stratify=df[col_estrato], random_state=semilla)
    return muestra.reset_index(drop=True)


def categorias_nuevas_en_prueba(X_train: pd.DataFrame, X_test: pd.DataFrame,
                                cat_cols: list[str]) -> dict[str, set]:
    """Categorias que aparecen en prueba pero no en entrenamiento (el encoder las maneja sin fallar)."""
    return {c: set(X_test[c].dropna().unique()) - set(X_train[c].dropna().unique()) for c in cat_cols}
