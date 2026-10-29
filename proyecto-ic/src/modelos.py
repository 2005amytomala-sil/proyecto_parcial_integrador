"""Comparacion de modelos (Fase III, Persona A).

Validacion cruzada estratificada SOLO sobre entrenamiento. El conjunto de prueba no se usa aqui.
Cada pliegue clona el Pipeline completo (preprocesamiento + modelo), asi que el escalado y la
codificacion se ajustan unicamente con los datos de entrenamiento de ese pliegue.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

SEED = 42
N_FOLDS = 5

# Hiperparametros que se muestran en la tabla de resultados
CLAVES_HP = {
    "Dummy (línea base)": ["strategy"],
    "KNN": ["n_neighbors", "weights", "p"],
    "SVM": ["kernel", "C", "gamma"],
    "Random Forest": ["n_estimators", "max_depth", "min_samples_leaf"],
    "MLP": ["hidden_layer_sizes", "activation", "alpha", "learning_rate_init", "early_stopping", "max_iter"],
}


def definir_modelos(preprocesador) -> dict[str, Pipeline]:
    """Un Pipeline por modelo, con configuracion inicial razonable y la misma semilla."""
    base = {
        "Dummy (línea base)": DummyClassifier(strategy="most_frequent"),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="uniform", p=2, n_jobs=-1),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=1,
                                                n_jobs=-1, random_state=SEED),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4,
                             learning_rate_init=1e-3, early_stopping=True, max_iter=200,
                             random_state=SEED),
    }
    return {n: Pipeline([("pre", clone(preprocesador)), ("modelo", m)]) for n, m in base.items()}


def descripcion_hiperparametros(nombre: str, pipeline: Pipeline) -> str:
    p = pipeline.named_steps["modelo"].get_params()
    return ", ".join(f"{k}={p[k]}" for k in CLAVES_HP[nombre])


def _puntaje_clase_positiva(pipeline: Pipeline, X) -> np.ndarray:
    """Probabilidad (o distancia al hiperplano en SVM) de la clase ataque, para ROC-AUC."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    return pipeline.decision_function(X)


def evaluar_cv(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series,
               k: int = N_FOLDS, semilla: int = SEED) -> dict:
    """Validacion cruzada estratificada con metricas, tiempos y matriz de confusion acumulada."""
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=semilla)
    filas, matriz = [], np.zeros((2, 2), dtype=int)
    for i_tr, i_va in cv.split(X, y):
        modelo = clone(pipeline)
        t0 = time.perf_counter()
        modelo.fit(X.iloc[i_tr], y.iloc[i_tr])
        t_fit = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred = modelo.predict(X.iloc[i_va])
        t_inf = time.perf_counter() - t0
        score = _puntaje_clase_positiva(modelo, X.iloc[i_va])

        yv = y.iloc[i_va]
        matriz += confusion_matrix(yv, pred, labels=[0, 1])
        filas.append({
            "f1": f1_score(yv, pred, zero_division=0),
            "precision": precision_score(yv, pred, zero_division=0),
            "recall": recall_score(yv, pred, zero_division=0),
            "roc_auc": roc_auc_score(yv, score),
            "exactitud": accuracy_score(yv, pred),
            "t_entrenamiento_s": t_fit,
            "t_inferencia_ms_por_1000": t_inf / len(i_va) * 1000 * 1000,
        })
    df = pd.DataFrame(filas)
    return {"folds": df, "media": df.mean(), "desv": df.std(ddof=1), "matriz_confusion": matriz}


def tabla_resultados(modelos: dict[str, Pipeline], resultados: dict[str, dict]) -> pd.DataFrame:
    """Tabla unica: hiperparametros, metricas (media y desviacion) y tiempos."""
    filas = []
    for nombre, r in resultados.items():
        m, d = r["media"], r["desv"]
        fila = {"Modelo": nombre, "Hiperparámetros": descripcion_hiperparametros(nombre, modelos[nombre])}
        for col, etiqueta in [("f1", "F1"), ("precision", "Precisión"), ("recall", "Recall"),
                              ("roc_auc", "ROC-AUC"), ("exactitud", "Exactitud")]:
            fila[f"{etiqueta} (media)"] = m[col]
            fila[f"{etiqueta} (desv)"] = d[col]
        fila["Entrenamiento (s/pliegue)"] = m["t_entrenamiento_s"]
        fila["Inferencia (ms/1000 reg.)"] = m["t_inferencia_ms_por_1000"]
        filas.append(fila)
    return pd.DataFrame(filas)
