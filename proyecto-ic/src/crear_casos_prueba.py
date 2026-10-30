"""Crea 4 CSV pequeños para probar el prototipo (Fase IV) en data/casos_prueba/.

Uso (desde proyecto-ic):
    python src/crear_casos_prueba.py --modelo models/pipeline_final.joblib --test data/processed/test_completo.csv
Los casos 1 y 2 se eligen entre registros que el modelo clasifica bien (son demostrativos; el
rendimiento real se reporta con la prueba completa). El caso 3 es el limite; el 4 debe dar error.
"""
import argparse
from pathlib import Path

import joblib, numpy as np, pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--modelo", default="models/pipeline_final.joblib")
p.add_argument("--test", default="data/processed/test_completo.csv")
p.add_argument("--out", default="data/casos_prueba")
a = p.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

m = joblib.load(a.modelo)
df = pd.read_csv(a.test)
X = df.drop(columns=["label", "attack_cat"], errors="ignore")[list(m.feature_names_in_)]
prob = m.predict_proba(X)[:, 1]
pred = (prob >= 0.5).astype(int)
ok = pred == df["label"].values

def elegir(mascara, n=10):
    return df[mascara].sample(n, random_state=42)

normal = elegir(ok & (df["label"] == 0) & (prob < 0.20))
ataque = elegir(ok & (df["label"] == 1) & (prob > 0.95))
normal.to_csv(out / "caso1_normal.csv", index=False)
ataque.to_csv(out / "caso2_ataque.csv", index=False)

# Caso limite: un registro cuya probabilidad es cercana a 0.5 (el modelo duda) + categorias nunca vistas.
dudoso = df.iloc[[int(np.argmin(np.abs(prob - 0.5)))]].copy()
raro = ataque.iloc[[0]].copy()
raro["proto"], raro["service"], raro["state"] = "protocolo_nuevo", "servicio_nuevo", "estado_nuevo"
pd.concat([dudoso, raro]).to_csv(out / "caso3_limite.csv", index=False)

# Caso de error: falta una columna requerida (la app debe avisar, no fallar).
ataque.drop(columns=[list(m.feature_names_in_)[0]]).to_csv(out / "caso4_columna_faltante.csv", index=False)
print("Creados en", out, ":", sorted(f.name for f in out.glob("*.csv")))
print("Probabilidad de ataque del caso limite (dudoso):", round(float(prob[int(np.argmin(np.abs(prob - 0.5)))]), 3))
