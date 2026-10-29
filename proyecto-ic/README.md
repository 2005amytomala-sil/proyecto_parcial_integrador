# Proyecto integrador: Inteligencia Computacional

Proyecto académico de detección binaria de tráfico normal y ataques con el dataset
UNSW-NB15.

## Estructura

- `data/`: CSV originales y datos procesados.
- `notebooks/`: exploración y preparación de datos.
- `src/`: código reutilizable, incluido el generador de figuras del EDA.
- `reports/`: figuras y resúmenes para el informe.
- `models/`: modelos entrenados.
- `app/`: prototipo de la aplicación.

## Preparar el entorno (Windows)

Ejecuta estos comandos desde la carpeta `proyecto-ic`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Generar las figuras del EDA

En la copia actual, los archivos tienen nombres opuestos a su contenido: `UNSW_NB15_testing-set.csv`
tiene 175341 registros y se usa como entrenamiento; `UNSW_NB15_training-set.csv` tiene 82332
y se conserva como prueba. Verifica esta asignación si reemplazas los CSV.

Desde `proyecto-ic`, ejecuta:

```powershell
.\.venv\Scripts\python.exe .\src\eda_figuras.py `
  --train .\data\UNSW_NB15_testing-set.csv `
  --out .\reports\figuras
```

El script utiliza solo el archivo de entrenamiento y guarda cinco figuras PNG y
`resumen_datos.txt` en `reports/figuras`.