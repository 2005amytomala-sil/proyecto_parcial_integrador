"""Verifica models/pipeline_final.joblib sin abrir Streamlit.

Uso (desde proyecto-ic):
    python src/verificar_modelo.py --modelo models/pipeline_final.joblib --test data/processed/test_completo.csv
Guarda el informe en reports/verificacion_modelo.txt (pegalo en el chat para revisarlo).
"""
import argparse, io, platform, time, warnings
from contextlib import redirect_stdout
from pathlib import Path

import joblib, numpy as np, pandas as pd, sklearn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

NO_PRED = ["label", "attack_cat"]


def intentar(nombre, f):
    try:
        r = f()
        print(f"  [OK]   {nombre}: {r}")
    except Exception as e:  # noqa: BLE001
        print(f"  [FALLA] {nombre}: {type(e).__name__}: {str(e)[:200]}")


def main(a):
    warnings.filterwarnings("ignore")
    ruta = Path(a.modelo)
    print("== ENTORNO ==")
    print("python", platform.python_version(), "| sklearn", sklearn.__version__,
          "| pandas", pd.__version__, "| numpy", np.__version__, "| joblib", joblib.__version__)
    print(f"Modelo: {ruta} | {ruta.stat().st_size / 1e6:.1f} MB (GitHub bloquea >100 MB)")
    m = joblib.load(ruta)
    print("Tipo:", type(m).__name__, "| pasos:", list(getattr(m, "named_steps", {})))
    est = getattr(m, "named_steps", {}).get("modelo", m)
    print("Estimador:", est)
    print("classes_:", list(m.classes_))
    cols = list(m.feature_names_in_)
    print(f"Columnas de entrada esperadas: {len(cols)}")
    print("  contiene id/label/attack_cat?:", [c for c in ["id", "label", "attack_cat"] if c in cols])

    df = pd.read_csv(a.test)
    print(f"\n== PRUEBA COMPLETA ({len(df)} filas) ==")
    y = df["label"].astype(int)
    X = df.drop(columns=[c for c in NO_PRED if c in df.columns])[cols]
    t0 = time.perf_counter()
    prob = m.predict_proba(X)[:, list(m.classes_).index(1)]
    seg = time.perf_counter() - t0
    pred = m.predict(X)
    print(f"Inferencia: {seg:.1f} s ({seg / len(X) * 1000:.3f} ms/registro)")
    print(f"Real: {int(y.sum())} ataques ({y.mean():.2%}) | Predicho: {int(pred.sum())} ataques ({pred.mean():.2%})")
    print(f"accuracy={accuracy_score(y, pred):.4f} precision={precision_score(y, pred):.4f} "
          f"recall={recall_score(y, pred):.4f} F1={f1_score(y, pred):.4f} ROC-AUC={roc_auc_score(y, prob):.4f}")
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    print(f"Matriz: VN={tn} FP={fp} FN={fn} VP={tp}  (tasa de falsos positivos {fp / (fp + tn):.2%})")
    if "attack_cat" in df.columns:
        print("\nRecall de deteccion por categoria (porcentaje detectado como ataque):")
        r = pd.Series(pred, index=df.index).groupby(df["attack_cat"]).mean().mul(100).round(1)
        print(r.sort_values().to_string())
    claridad = np.maximum(prob, 1 - prob)
    print("\nNiveles de confianza con umbrales de la app (0.65 / 0.85):")
    niv = pd.cut(claridad, [0, .65, .85, 1.01], labels=["Baja", "Media", "Alta"], right=False)
    print(pd.Series(niv).value_counts().reindex(["Alta", "Media", "Baja"]).to_string())

    print("\n== CASOS DE PRUEBA (como en el prototipo) ==")
    normal = X[y == 0].iloc[[0]]
    ataque = X[y == 1].iloc[[0]]
    print("  Normal  (real 0): prob_ataque =", round(float(m.predict_proba(normal)[0, 1]), 3))
    print("  Ataque  (real 1): prob_ataque =", round(float(m.predict_proba(ataque)[0, 1]), 3))
    raro = ataque.copy()
    raro["proto"], raro["service"], raro["state"] = "protocolo_nuevo", "servicio_nuevo", "estado_nuevo"
    intentar("limite: categorias nunca vistas", lambda: round(float(m.predict_proba(raro)[0, 1]), 3))
    nulo = ataque.copy(); nulo.iloc[0, 1] = np.nan
    intentar("limite: valor faltante (NaN)", lambda: round(float(m.predict_proba(nulo)[0, 1]), 3))
    extremo = ataque.copy(); extremo["sbytes"] = 1e12; extremo["dur"] = 1e6
    intentar("limite: valores extremos", lambda: round(float(m.predict_proba(extremo)[0, 1]), 3))
    sin_col = ataque.drop(columns=[cols[0]])
    intentar("falta una columna (debe FALLAR; la app lo avisa antes)", lambda: m.predict(sin_col))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", default="models/pipeline_final.joblib")
    p.add_argument("--test", default="data/processed/test_completo.csv")
    p.add_argument("--out", default="reports/verificacion_modelo.txt")
    a = p.parse_args()
    buf = io.StringIO()
    class Tee(io.TextIOBase):
        def write(self, s): print(s, end="", file=__import__("sys").__stdout__); buf.write(s); return len(s)
    with redirect_stdout(Tee()):
        main(a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(buf.getvalue(), encoding="utf-8")
