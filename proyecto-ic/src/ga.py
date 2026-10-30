"""Algoritmo genetico para optimizar hiperparametros de Random Forest (Fase III, Persona B).

Todo se evalua con validacion cruzada estratificada SOLO sobre el entrenamiento (la submuestra
generada en 01_eda). El conjunto de prueba no se usa aqui. Cada evaluacion clona el Pipeline
completo (preprocesamiento + modelo), asi que el escalado y la codificacion se ajustan unicamente
con los datos de entrenamiento de cada pliegue.

Piezas:
- ESPACIO_RF / CONFIG_INICIAL: espacio discreto de busqueda y la configuracion de 02_modelos.
- EvaluadorRF: funcion de aptitud (F1 de la clase ataque, media en k pliegues) con cache y registro.
- algoritmo_genetico: poblacion, seleccion por torneo, cruce uniforme, mutacion y elitismo.
- busqueda_aleatoria: referencia con el mismo presupuesto de evaluaciones y la misma aptitud.
- construir_pipeline_rf: arma el Pipeline final con una configuracion (se reutiliza en la Fase IV).
"""
from __future__ import annotations

import random
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

SEED = 42

# Espacio de busqueda: 11 x 6 x 6 x 4 = 1584 combinaciones.
ESPACIO_RF: dict[str, list] = {
    "n_estimators": list(range(50, 301, 25)),
    "max_depth": [5, 10, 15, 20, 30, None],
    "min_samples_leaf": [1, 2, 3, 5, 8, 12],
    "max_features": ["sqrt", "log2", 0.3, 0.5],
}

# Configuracion inicial de 02_modelos (max_features="sqrt" es el valor por defecto del clasificador).
CONFIG_INICIAL: dict = {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"}


def tamano_espacio(espacio: dict[str, list] = ESPACIO_RF) -> int:
    return int(np.prod([len(v) for v in espacio.values()]))


def construir_pipeline_rf(preprocesador, config: dict, semilla: int = SEED) -> Pipeline:
    """Pipeline (preprocesamiento + Random Forest) para una configuracion de hiperparametros."""
    modelo = RandomForestClassifier(**config, n_jobs=-1, random_state=semilla)
    return Pipeline([("pre", clone(preprocesador)), ("modelo", modelo)])


def describir(config: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in config.items())


class EvaluadorRF:
    """Aptitud = F1 (clase ataque) medio en k pliegues estratificados, solo con entrenamiento.

    Guarda en cache las configuraciones ya evaluadas (no se repite un entrenamiento) y un
    registro con el orden de las evaluaciones, util para comparar contra la busqueda aleatoria.
    """

    def __init__(self, preprocesador, X: pd.DataFrame, y: pd.Series, k: int = 3,
                 semilla: int = SEED, espacio: dict[str, list] = ESPACIO_RF):
        self.preprocesador, self.X, self.y = preprocesador, X, y
        self.espacio = espacio
        self.cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=semilla)
        self.semilla = semilla
        self._cache: dict[tuple, float] = {}
        self.registro: list[dict] = []
        self._t0 = time.perf_counter()

    @property
    def n_evaluaciones(self) -> int:
        """Entrenamientos distintos realizados (las repeticiones salen de la cache)."""
        return len(self.registro)

    def __call__(self, config: dict) -> float:
        clave = tuple(config[g] for g in self.espacio)
        if clave in self._cache:
            return self._cache[clave]
        t0 = time.perf_counter()
        pipe = construir_pipeline_rf(self.preprocesador, config, self.semilla)
        puntajes = cross_val_score(pipe, self.X, self.y, cv=self.cv, scoring="f1")
        aptitud = float(np.mean(puntajes))
        self._cache[clave] = aptitud
        self.registro.append({
            "evaluacion": len(self.registro) + 1, **config,
            "aptitud": aptitud, "desv": float(np.std(puntajes, ddof=1)),
            "segundos": time.perf_counter() - t0,
            "t_acumulado_s": time.perf_counter() - self._t0,
        })
        return aptitud

    def tabla_registro(self) -> pd.DataFrame:
        df = pd.DataFrame(self.registro)
        if len(df):
            df["mejor_hasta_ahora"] = df["aptitud"].cummax()
        return df


# --------------------------------------------------------------------------- operadores
def _individuo_aleatorio(espacio: dict[str, list], rng: random.Random) -> dict:
    return {g: rng.choice(valores) for g, valores in espacio.items()}


def _torneo(poblacion: list[dict], aptitudes: list[float], k: int, rng: random.Random) -> dict:
    """Elige k individuos al azar y devuelve el de mayor aptitud."""
    candidatos = rng.sample(range(len(poblacion)), min(k, len(poblacion)))
    return poblacion[max(candidatos, key=lambda i: aptitudes[i])]


def _cruce_uniforme(p1: dict, p2: dict, prob: float, rng: random.Random) -> tuple[dict, dict]:
    """Cada gen del hijo viene de uno de los dos padres con probabilidad 0,5."""
    if rng.random() > prob:
        return dict(p1), dict(p2)
    h1, h2 = {}, {}
    for gen in p1:
        if rng.random() < 0.5:
            h1[gen], h2[gen] = p1[gen], p2[gen]
        else:
            h1[gen], h2[gen] = p2[gen], p1[gen]
    return h1, h2


def _mutar(individuo: dict, espacio: dict[str, list], prob: float, rng: random.Random) -> dict:
    """Con probabilidad `prob` por gen, lo reemplaza por otro valor permitido al azar."""
    nuevo = dict(individuo)
    for gen, valores in espacio.items():
        if rng.random() < prob:
            nuevo[gen] = rng.choice(valores)
    return nuevo


# --------------------------------------------------------------------------- algoritmo genetico
def algoritmo_genetico(evaluar: EvaluadorRF, espacio: dict[str, list] = ESPACIO_RF,
                       tam_poblacion: int = 8, generaciones: int = 5,
                       prob_cruce: float = 0.8, prob_mutacion: float = 0.25,
                       k_torneo: int = 3, elitismo: int = 1, paciencia: int | None = None,
                       semilla: int = SEED, verbose: bool = True) -> dict:
    """Maximiza la aptitud de `evaluar`. La generacion 0 es la poblacion aleatoria inicial.

    Presupuesto maximo: tam_poblacion * (generaciones + 1) evaluaciones; menos si hay repetidas.
    """
    rng = random.Random(semilla)
    t0 = time.perf_counter()
    poblacion = [_individuo_aleatorio(espacio, rng) for _ in range(tam_poblacion)]
    historia, mejor, mejor_aptitud, sin_mejora = [], None, -np.inf, 0

    for gen in range(generaciones + 1):
        aptitudes = [evaluar(ind) for ind in poblacion]
        i_mejor = int(np.argmax(aptitudes))
        if aptitudes[i_mejor] > mejor_aptitud + 1e-12:
            mejor, mejor_aptitud, sin_mejora = dict(poblacion[i_mejor]), aptitudes[i_mejor], 0
        else:
            sin_mejora += 1
        historia.append({"generacion": gen, "mejor": max(aptitudes), "media": float(np.mean(aptitudes)),
                         "peor": min(aptitudes), "mejor_global": mejor_aptitud,
                         "evaluaciones": evaluar.n_evaluaciones, "tiempo_s": time.perf_counter() - t0})
        if verbose:
            h = historia[-1]
            print(f"gen {gen}: mejor={h['mejor']:.4f} media={h['media']:.4f} "
                  f"| mejor global={mejor_aptitud:.4f} | evaluaciones={h['evaluaciones']} | {h['tiempo_s']:.0f} s")
        if gen == generaciones or (paciencia is not None and sin_mejora >= paciencia):
            break

        orden = np.argsort(aptitudes)[::-1]
        nueva = [dict(poblacion[i]) for i in orden[:elitismo]]          # elitismo
        while len(nueva) < tam_poblacion:
            p1 = _torneo(poblacion, aptitudes, k_torneo, rng)
            p2 = _torneo(poblacion, aptitudes, k_torneo, rng)
            for hijo in _cruce_uniforme(p1, p2, prob_cruce, rng):
                nueva.append(_mutar(hijo, espacio, prob_mutacion, rng))
                if len(nueva) == tam_poblacion:
                    break
        poblacion = nueva

    return {"mejor_config": mejor, "mejor_aptitud": mejor_aptitud,
            "historia": pd.DataFrame(historia), "evaluaciones": evaluar.n_evaluaciones,
            "tiempo_s": time.perf_counter() - t0}


# --------------------------------------------------------------------------- referencia
def busqueda_aleatoria(evaluar: EvaluadorRF, n_evaluaciones: int,
                       espacio: dict[str, list] = ESPACIO_RF, semilla: int = SEED,
                       verbose: bool = True) -> dict:
    """Muestrea configuraciones distintas al azar con la MISMA aptitud y el mismo presupuesto.

    Usa un EvaluadorRF propio (misma semilla de pliegues que el del GA), por lo que ambos metodos
    se miden exactamente igual.
    """
    rng = random.Random(semilla + 1)
    t0 = time.perf_counter()
    intentos, maximo = 0, n_evaluaciones * 50
    while evaluar.n_evaluaciones < n_evaluaciones and intentos < maximo:
        intentos += 1
        evaluar(_individuo_aleatorio(espacio, rng))
    tabla = evaluar.tabla_registro()
    fila = tabla.loc[tabla["aptitud"].idxmax()]
    mejor = {g: (None if pd.isna(fila[g]) else fila[g]) for g in espacio}
    for g in ("n_estimators", "min_samples_leaf"):           # pandas devuelve numpy; los dejamos como int
        mejor[g] = int(mejor[g])
    if mejor.get("max_depth") is not None:
        mejor["max_depth"] = int(mejor["max_depth"])
    if isinstance(mejor.get("max_features"), float) and mejor["max_features"] not in espacio["max_features"]:
        mejor["max_features"] = float(mejor["max_features"])
    if verbose:
        print(f"busqueda aleatoria: mejor={fila['aptitud']:.4f} | evaluaciones={evaluar.n_evaluaciones} "
              f"| {time.perf_counter() - t0:.0f} s")
    return {"mejor_config": mejor, "mejor_aptitud": float(fila["aptitud"]), "tabla": tabla,
            "evaluaciones": evaluar.n_evaluaciones, "tiempo_s": time.perf_counter() - t0}
