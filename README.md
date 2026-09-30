# Pipeline MLOps y Seguimiento de Experimentos con MLflow: Predicción de Calidad de Vinos

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-v2.x-0194E2.svg)](https://mlflow.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Este proyecto implementa un pipeline integral y reproducible de **MLOps** centrado en el **seguimiento de experimentos, registro de artefactos y gobernanza de modelos (Model Registry)** utilizando **MLflow** como herramienta principal, contrastado empíricamente contra **Weights & Biases (W&B)**.

---

## 👥 Integrantes y Roles

* **Joseph Salamea:**
  * Análisis Exploratorio de Datos (EDA), limpieza y preprocesamiento de los conjuntos de datos de vino (tinto y blanco).
  * Implementación del seguimiento comparativo en **Weights & Biases (W&B)**.
  * Medición y evaluación empírica de los criterios de comparación (latencia de registro y autonomía offline).
  * Coautoría de la documentación técnica y material educativo.

* **Xavier Peña:**
  * Diseño, configuración y ejecución de la suite de 10+ experimentos en **MLflow Tracking**.
  * Búsqueda de hiperparámetros y optimización multimodelo (Regresión Logística, Random Forest, XGBoost).
  * Empaquetado, versionado y registro del modelo campeón en el **MLflow Model Registry** con transiciones de estado (`Staging` / `Production`).
  * Coautoría de la documentación técnica y material educativo.

---

## 🎯 Objetivos del Proyecto

1. **Predicción y Clasificación de Calidad:** Evaluar la calidad del vino portugués *"Vinho Verde"* (en variantes tinto y blanco, 6.497 instancias combinadas) en función de 11 variables fisicoquímicas continuas.
2. **Trazabilidad y MLOps con MLflow:** Registrar sistemáticamente al menos 10 corridas de experimentos variando modelos e hiperparámetros, guardando parámetros, métricas (`F1-Score`, `Accuracy`, `ROC-AUC`, `RMSE`), matrices de confusión y artefactos.
3. **Gobernanza y Reproducibilidad:** Versionar datos y modelos de forma determinista, promoviendo el mejor modelo al **Model Registry** de MLflow.
4. **Comparación Empírica:** Medir cuantitativamente:
   * **Criterio 1 (Sobrecarga / Latencia de registro):** Overhead en milisegundos del registro local en MLflow vs. registro remoto en la nube con W&B.
   * **Criterio 2 (Autonomía e infraestructura):** Capacidad de operar 100% desconectado y sin cuentas (MLflow) vs. dependencia estricta de internet y credenciales de usuario (W&B).

---

## 📂 Estructura del Repositorio

```text
├── data/                               # Conjuntos de datos locales (< 1 MB)
│   ├── winequality-red.csv             # 1.599 instancias de vino tinto
│   ├── winequality-white.csv           # 4.898 instancias de vino blanco
│   └── winequality.names               # Documentación y metadatos del dataset
├── docs/                               # Documentación e investigación técnica
│   └── investigacion_mlflow.md         # Marco teórico, arquitectura, comparativa y APA 7
├── notebooks/                          # Cuadernos reproducibles
│   └── pipeline_winequality_mlflow.ipynb # Pipeline completo: EDA, MLflow, W&B y Métricas
├── .gitignore                          # Exclusión de entornos, mlruns, wandb y caches
├── requirements.txt                    # Dependencias y versiones fijadas del entorno
└── README.md                           # Descripción del proyecto, roles y ejecución
```

---

## 🚀 Instrucciones de Instalación y Ejecución

### 1. Clonar el Repositorio
```bash
git clone https://github.com/<USUARIO_O_ORGANIZACION>/mlops-winequality-mlflow.git
cd mlops-winequality-mlflow
```

### 2. Crear y Activar un Entorno Virtual (Recomendado)
```bash
# En Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# En Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar Dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Ejecutar el Pipeline Reproducible
Puedes abrir y ejecutar el cuaderno interactivo en Jupyter Lab o VS Code:
```bash
jupyter lab notebooks/pipeline_winequality_mlflow.ipynb
```
*El notebook está diseñado para ejecutarse de principio a fin de forma determinista (fijando semillas aleatorias `random_state=42`).*

### 5. Lanzar la Interfaz de Usuario de MLflow
Para visualizar localmente los 10+ experimentos, curvas y el Model Registry:
```bash
mlflow ui
```
Luego abre tu navegador en: [http://localhost:5000](http://localhost:5000).

---

## 📊 Dataset Utilizado

* **Nombre:** Wine Quality Dataset (UCI Machine Learning Repository, ID 186).
* **Autores:** Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties*. Decision Support Systems, 47(4), 547–553.
* **Licencia:** Creative Commons Attribution 4.0 International (CC BY 4.0).
* **Variables Predictoras (11):** Acidez fija, acidez volátil, ácido cítrico, azúcar residual, cloruros, dióxido de azufre libre, dióxido de azufre total, densidad, pH, sulfatos, alcohol.
* **Variable Objetivo:** Calidad (`quality`), puntuación sensorial de 0 a 10.

---

## 🤖 Declaración de Uso de Inteligencia Artificial Generativa

En cumplimiento con la política de integridad académica y uso responsable de IA de la asignatura:

1. **Herramientas de IA Utilizadas:** Claude 3.7 / Antigravity AI Assistant.
2. **Tareas Realizadas con IA:**
   * Apoyo en la estructuración de la documentación técnica y referencias bibliográficas.
   * Asistencia en la generación de plantillas de código para el registro de métricas en MLflow y Weights & Biases.
   * Revisión de sintaxis en consultas de visualización con Matplotlib/Seaborn.
3. **Método de Verificación Humana:**
   * Cada fragmento de código fue revisado, adaptado y validado experimentalmente por los integrantes en el entorno local.
   * Las definiciones técnicas y citas académicas se contrastaron directamente con los artículos originales (Zaharia et al., 2018; Cortez et al., 2009; Sculley et al., 2015) y la documentación oficial de MLflow v2.x.
   * Todos los experimentos y mediciones de tiempo fueron ejecutados sobre la máquina local, garantizando resultados empíricos verídicos y reproducibles.
