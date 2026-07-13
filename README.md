# TFG — Explainable ML for LaLiga Match Prediction

Predicción explicable de resultados de partidos de LaLiga (2005–2025) usando únicamente
datos pre-partido. Arquitectura en cascada: un modelo binario (1 vs X2) y, si el
resultado no es victoria local, un segundo modelo decide entre empate y victoria
visitante (X vs 2). Las predicciones se acompañan de explicaciones SHAP.

## Stack tecnológico

- **Python** — pandas, NumPy, scikit-learn
- **Modelos** — Random Forest, XGBoost, CatBoost, TabPFN (con wrapper de balanced bagging)
- **Explicabilidad** — SHAP, LIME, permutation importance
- **Incertidumbre** — MAPIE (conformal prediction)
- **Despliegue** — MLflow (registro de modelos), FastAPI (API), Streamlit (interfaz)

## Ejecución

Los tres servicios deben arrancarse **en este orden**, cada uno en su propia terminal.

**1. Servidor MLflow** (puerto 5000)
```bash
mlflow server --host 0.0.0.0 --port 5000
```

**2. Registrar los modelos** (solo la primera vez)
```bash
python mlflow/mlflow_model_load.py
```

**3. API** (puerto 8001)
```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8001
```

**4. Aplicación Streamlit** (puerto 8501)
```bash
streamlit run app/Home.py
```

La aplicación estará disponible en `http://localhost:8501`.

## Configuración

Crear un archivo `.env` en la raíz del proyecto con el token de TabPFN: