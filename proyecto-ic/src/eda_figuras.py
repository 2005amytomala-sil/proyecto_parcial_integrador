"""Figuras del EDA de UNSW-NB15 para el informe (Fase II).

Usa SOLO el conjunto de entrenamiento: el de prueba no se toca en el EDA,
para no tomar decisiones de preprocesamiento a partir de él.

Uso:
    python src/eda_figuras.py --train data/UNSW_NB15_testing-set.csv --out reports/figuras

En esta copia local, ese archivo contiene los 175341 registros del entrenamiento.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: guarda PNG directamente
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SEED = 42
np.random.seed(SEED)

COLUMNAS_EXCLUIDAS = ["id", "attack_cat"]  # identificador y fuga de información
CATEGORICAS = ["proto", "service", "state"]
VARIABLES_CLAVE = ["dur", "sbytes", "dbytes", "sttl", "sload", "ct_state_ttl", "smean", "rate"]
ETIQUETAS = {0: "Normal", 1: "Ataque"}
sns.set_theme(style="whitegrid")


def guardar(fig, salida: Path, nombre: str) -> None:
    ruta = salida / nombre
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  guardada: {ruta}")


def resumen_texto(df: pd.DataFrame, salida: Path) -> None:
    """Datos para la sección de calidad de datos del informe."""
    sin_id = df.drop(columns=[c for c in ["id"] if c in df.columns])
    lineas = [
        f"Filas: {len(df)}  Columnas: {df.shape[1]}",
        f"Valores faltantes totales: {int(df.isna().sum().sum())}",
        f"Registros duplicados (sin contar id): {int(sin_id.duplicated().sum())}",
        "",
        "Distribución de la variable objetivo (label):",
        df["label"].map(ETIQUETAS).value_counts().to_string(),
        "",
        "Proporción:",
        (df["label"].map(ETIQUETAS).value_counts(normalize=True) * 100).round(2).to_string(),
        "",
        "Tipos de dato:",
        df.dtypes.astype(str).value_counts().to_string(),
    ]
    if "attack_cat" in df.columns:
        lineas += ["", "Categorías de ataque:", df["attack_cat"].value_counts().to_string()]
    (salida / "resumen_datos.txt").write_text("\n".join(lineas), encoding="utf-8")
    print("  guardado:", salida / "resumen_datos.txt")


def fig_balance_clases(df: pd.DataFrame, salida: Path) -> None:
    conteo = df["label"].map(ETIQUETAS).value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x=conteo.index, y=conteo.values, ax=ax)
    for i, valor in enumerate(conteo.values):
        ax.text(i, valor, f"{valor:,}\n({valor / conteo.sum():.1%})", ha="center", va="bottom")
    ax.set_title("Distribución de la variable objetivo (entrenamiento)")
    ax.set_xlabel("")
    ax.set_ylabel("Registros")
    ax.set_ylim(0, conteo.max() * 1.2)
    guardar(fig, salida, "fig1_balance_clases.png")


def fig_categorias_ataque(df: pd.DataFrame, salida: Path) -> None:
    if "attack_cat" not in df.columns:
        return
    conteo = df["attack_cat"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(x=conteo.values, y=conteo.index, ax=ax)
    ax.set_xscale("log")
    ax.set_title("Registros por categoría (escala logarítmica)")
    ax.set_xlabel("Registros")
    ax.set_ylabel("")
    guardar(fig, salida, "fig2_categorias_ataque.png")


def fig_distribuciones(df: pd.DataFrame, salida: Path) -> None:
    variables = [v for v in VARIABLES_CLAVE if v in df.columns]
    columnas = 4
    filas = int(np.ceil(len(variables) / columnas))
    fig, ejes = plt.subplots(filas, columnas, figsize=(4 * columnas, 3.2 * filas))
    ejes = np.atleast_1d(ejes).ravel()
    datos = df.assign(clase=df["label"].map(ETIQUETAS))
    for eje, variable in zip(ejes, variables):
        datos["_x"] = np.log1p(datos[variable].clip(lower=0))
        sns.histplot(data=datos, x="_x", hue="clase", bins=40, stat="density",
                     common_norm=False, element="step", ax=eje)
        eje.set_title(variable)
        eje.set_xlabel("log(1 + valor)")
    for eje in ejes[len(variables):]:
        eje.axis("off")
    fig.suptitle("Distribución de variables clave: normal vs ataque", y=1.02)
    guardar(fig, salida, "fig3_distribuciones_clave.png")


def fig_categoricas(df: pd.DataFrame, salida: Path) -> None:
    presentes = [c for c in CATEGORICAS if c in df.columns]
    fig, ejes = plt.subplots(1, len(presentes), figsize=(5.5 * len(presentes), 4.5))
    ejes = np.atleast_1d(ejes)
    for eje, columna in zip(ejes, presentes):
        top = df[columna].value_counts().head(8).index
        tasa = (df[df[columna].isin(top)].groupby(columna)["label"].mean()
                .sort_values(ascending=False))
        sns.barplot(x=tasa.values, y=tasa.index, ax=eje)
        eje.set_xlim(0, 1)
        eje.set_title(f"Proporción de ataque por {columna} (8 más frecuentes)")
        eje.set_xlabel("Proporción de registros de ataque")
        eje.set_ylabel("")
    guardar(fig, salida, "fig4_categoricas_vs_ataque.png")


def fig_correlacion(df: pd.DataFrame, salida: Path) -> None:
    numericas = df.drop(columns=COLUMNAS_EXCLUIDAS, errors="ignore").select_dtypes("number")
    corr_label = numericas.corr()["label"].drop("label").sort_values(key=abs, ascending=False)
    top = corr_label.head(12).index.tolist()

    fig, ejes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.barplot(x=corr_label.head(15).values, y=corr_label.head(15).index, ax=ejes[0])
    ejes[0].set_title("Correlación de cada variable con label (15 mayores)")
    ejes[0].set_xlabel("Correlación de Pearson")
    sns.heatmap(numericas[top + ["label"]].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ejes[1], annot_kws={"size": 7})
    ejes[1].set_title("Matriz de correlación (12 variables más ligadas a label)")
    guardar(fig, salida, "fig5_correlacion.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Figuras del EDA (solo entrenamiento)")
    parser.add_argument("--train", required=True, help="Ruta a UNSW_NB15_training-set.csv")
    parser.add_argument("--out", default="reports/figuras", help="Carpeta de salida")
    args = parser.parse_args()

    salida = Path(args.out)
    salida.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.train)
    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

    resumen_texto(df, salida)
    fig_balance_clases(df, salida)
    fig_categorias_ataque(df, salida)
    fig_distribuciones(df, salida)
    fig_categoricas(df, salida)
    fig_correlacion(df, salida)
    print("Listo.")


if __name__ == "__main__":
    main()
