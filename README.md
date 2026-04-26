# Taller 3 — Diseño y Optimización de un Perceptrón Multicapa (MLP)

Diseño, entrenamiento y optimización de una red neuronal MLP completamente densa
para reconocimiento de identidad facial sobre el dataset **Olivetti Faces**,
usando únicamente `scikit-learn` (sin redes convolucionales).

---

## Objetivo

Construir un MLP que cumpla los siguientes requisitos de diseño:

- Completamente denso (Fully Connected), sin capas convolucionales.
- Al menos dos capas ocultas.
- Capa de salida con activación Softmax.
- Justificación de: número de capas, número de neuronas, funciones de activación,
  función de pérdida y optimizador.

---

## Dataset: Olivetti Faces

| Característica | Valor |
|---|---|
| Total de imágenes | 400 |
| Personas distintas | 40 |
| Imágenes por persona | 10 |
| Resolución | 64 x 64 píxeles (escala de grises) |
| Representación de entrada | Vector plano de 4096 valores |
| Clases de salida | 40 (una por persona) |

Cada imagen se aplana a un vector de 4096 valores y se normaliza al rango `[0, 1]`
dividiendo entre 255.

### Por qué el problema es difícil

La red tiene que aprender a distinguir 40 identidades distintas contando con
**solo 8 fotos de entrenamiento por persona**. Eso es un caso clásico de
*Small Data* con alta dimensionalidad (4096 variables de entrada). Si la red
memoriza las fotos en lugar de aprender patrones generales, falla ante cualquier
foto nueva — ese fenómeno se llama **overfitting**.

---

## Estructura del Proyecto

```
Taller3/
├── Taller3.ipynb           # Notebook principal con todo el desarrollo
├── olivetti_faces.npy      # Dataset en formato NumPy (ver nota abajo)
├── requirements.txt        # Dependencias con versiones exactas
└── README.md               # Este archivo
```

---

## Requisitos del Sistema

| Componente | Version probada |
|---|---|
| Python | 3.9 o superior (probado en 3.13.3) |
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| matplotlib | 3.10.9 |
| scikit-learn | 1.8.0 |
| ipykernel | 6.0 o superior |

El notebook fue desarrollado y verificado en **Windows 11** con Python 3.13.3.
Funciona igualmente en macOS y Linux con las mismas versiones de paquetes.

---

## Instalacion paso a paso

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd Taller3
```

### 2. Crear un entorno virtual (recomendado)

Usar un entorno virtual evita conflictos con otros proyectos instalados en el
sistema.

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar las dependencias

Con el entorno virtual activo:

```bash
pip install -r requirements.txt
```

Esto instala exactamente las mismas versiones usadas durante el desarrollo,
garantizando resultados reproducibles.

Si se prefiere instalar manualmente:

```bash
pip install numpy==2.4.4 pandas==3.0.2 matplotlib==3.10.9 scikit-learn==1.8.0 ipykernel
```

### 4. Obtener el dataset

El archivo `olivetti_faces.npy` no se incluye en el repositorio por su tamaño.
Generarlo toma menos de 10 segundos con scikit-learn, que lo descarga
automáticamente desde su servidor:

```python
from sklearn.datasets import fetch_olivetti_faces
import numpy as np

data = fetch_olivetti_faces()
np.save("olivetti_faces.npy", data.images)
```

Ejecutar ese bloque una sola vez en la misma carpeta del notebook. El archivo
resultante pesa aproximadamente 12 MB.

### 5. Abrir y ejecutar el notebook

**Opcion A — VS Code:**
1. Abrir VS Code en la carpeta del proyecto.
2. Instalar la extension **Jupyter** si no esta instalada.
3. Abrir `Taller3.ipynb`.
4. Seleccionar el kernel del entorno virtual (`.venv`).
5. Hacer clic en **Run All** (o `Ctrl+Alt+R`).

**Opcion B — Jupyter Lab / Notebook clasico:**
```bash
pip install jupyterlab
jupyter lab
```
Luego abrir `Taller3.ipynb` desde la interfaz y ejecutar **Run All Cells**.

**Opcion C — Linea de comandos (sin interfaz grafica):**
```bash
pip install nbconvert nbclient
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=900 Taller3.ipynb --output Taller3_ejecutado.ipynb
```

---

## Tiempo de ejecucion estimado

| Celda | Descripcion | Tiempo aproximado |
|---|---|---|
| Celda 1 | Carga y preprocesamiento | < 5 segundos |
| Celda 3 | RandomizedSearchCV (15 combinaciones x 3 folds) | 5 a 10 minutos |
| Celda 5 | Re-entrenamiento de los 3 mejores modelos | 3 a 6 minutos |
| Celda 7 | Evaluacion final y graficas | < 30 segundos |
| Celda 9 | Visualizacion de rostros | < 5 segundos |

El tiempo total depende del hardware. En un equipo con CPU de 4 nucleos o mas,
el proceso completo tarda entre **8 y 15 minutos**.

---

## Reproducibilidad

Todos los procesos aleatorios del notebook usan `random_state=42` y
`np.random.seed(42)`, lo que garantiza que los resultados sean identicos en
cualquier maquina con las mismas versiones de paquetes.

Los resultados esperados son:

| Metrica | Valor |
|---|---|
| Accuracy en prueba (Top 1) | ~92.5% |
| Accuracy en entrenamiento | ~99.4% |
| Brecha de overfitting | ~6.9% |

---

## Desarrollo del Notebook

### Parte 1 — Preprocesamiento

- Carga del dataset desde `olivetti_faces.npy`.
- Aplanado de imagenes `(400, 64, 64)` a vectores `(400, 4096)`.
- Normalizacion al rango `[0, 1]`.
- Division estratificada **80 / 10 / 10** (entrenamiento / validacion / prueba).

La estratificacion es obligatoria: garantiza que cada una de las 40 personas
tenga exactamente **8 fotos en entrenamiento, 1 en validacion y 1 en prueba**.
Sin ella, una persona podria quedar sin representacion en algun conjunto.

### Parte 2 — Diseño del MLP Base

Arquitectura de embudo: `4096 → 512 → 256 → 40`

| Componente | Eleccion | Justificacion |
|---|---|---|
| Capas ocultas | 2 (512 y 256 neuronas) | Aprendizaje jerarquico: la primera capa detecta bordes y texturas, la segunda combina esos patrones en geometrias faciales |
| Activacion ocultas | ReLU | Evita el desvanecimiento del gradiente y acelera la convergencia al no saturar en valores positivos |
| Activacion salida | Softmax | Convierte las 40 salidas en probabilidades que suman 1, necesario para clasificacion multiclase |
| Funcion de perdida | Categorical Crossentropy (Log-Loss) | Matematicamente compatible con Softmax; penaliza fuertemente las predicciones incorrectas con alta confianza |
| Optimizador | Adam | Combina RMSProp y Momentum; adapta la tasa de aprendizaje por peso, convergencia mas rapida y robusta |

### Parte 3 — Busqueda de Hiperparametros

Se usa `RandomizedSearchCV` con 15 combinaciones aleatorias y validacion cruzada
de 3 pliegues (3-fold CV). Es mas eficiente que Grid Search exhaustivo.

Espacio de busqueda explorado:

| Hiperparametro | Valores candidatos |
|---|---|
| Arquitectura (capas ocultas) | (512,256), (256,128), (512,256,128) |
| Funcion de activacion | relu, tanh |
| Optimizador | adam, sgd |
| Regularizacion L2 (alpha) | 0.0001, 0.01, 0.1 |
| Tasa de aprendizaje inicial | 0.001, 0.01 |
| Tamano de lote (batch size) | 32, 64 |
| Maximo de epocas | 300, 500 |

**Regularizacion aplicada:**
- **L2 (alpha):** penaliza pesos grandes, fuerza a la red a generalizar.
- **Early Stopping:** detiene el entrenamiento cuando el error de validacion
  deja de mejorar, evitando que la red memorice el ruido.

> `scikit-learn` no incluye Dropout nativo en `MLPClassifier`, por eso se
> combinan L2 y Early Stopping como mecanismos equivalentes.

### Parte 4 — Seleccion de los Tres Mejores Modelos

Se extraen los 3 modelos con mayor accuracy promedio en validacion cruzada.
Se presentan en una tabla comparativa y se grafican sus curvas de perdida
(Train Loss vs. Error de Validacion por epoca).

Las curvas muestran el descenso de la perdida durante el entrenamiento. El punto
donde Early Stopping detiene el ciclo es el momento de maxima eficiencia: la red
aprendio lo suficiente sin empezar a memorizar.

### Parte 5 — Evaluacion Final del Mejor Modelo

El modelo Top 1 se enfrenta al conjunto de prueba (`X_test`), que estuvo
completamente aislado durante todo el proceso.

**Metricas reportadas:**
- Accuracy global en prueba.
- Reporte de clasificacion por clase: Precision, Recall, F1-Score (tabla con
  gradiente de color por F1).
- Metricas globales de resumen: accuracy, macro avg, weighted avg.
- Matriz de Confusion (40x40, mapa de calor sin valores internos para no saturar).
- Analisis de overfitting: brecha entre accuracy de entrenamiento y de prueba.

**Visualizacion de predicciones — los 10 rostros:**

Se muestran 10 fotos del conjunto de prueba con el resultado de la red:
- Titulo en **verde**: la red acerto la identidad.
- Titulo en **rojo**: la red se equivoco.

El dataset tiene 40 personas con 10 fotos cada una. La red vio 8 fotos de cada
persona durante el entrenamiento. En esta etapa se le muestra la foto numero 9
(que nunca vio) y tiene que elegir a cual de las 40 personas pertenece. Los 10
rostros mostrados son una muestra aleatoria de ese examen final.

### Parte 6 — Discusion y Analisis Tecnico

Se responden cuatro preguntas clave:

1. **Impacto de la profundidad:** dos capas ocultas permiten aprendizaje
   jerarquico (bordes → geometrias faciales). Agregar mas capas sin datos
   suficientes no mejora el resultado.

2. **Necesidad de la regularizacion:** con solo 8 fotos por persona y 4096
   variables, el overfitting sin regularizacion es inevitable. L2 + Early
   Stopping lo controlan eficazmente.

3. **Overfitting observado:** la busqueda de hiperparametros redujo
   drasticamente la brecha entre accuracy de entrenamiento y de prueba respecto
   a los primeros prototipos.

4. **Limitaciones del MLP denso:** al aplanar la imagen a un vector 1D, la red
   pierde toda la estructura espacial 2D. Un desplazamiento de pocos pixeles
   produce una entrada completamente diferente. Esto motiva el uso de **Redes
   Convolucionales (CNN)** para este tipo de problemas en produccion real.

---

## Tecnologias

- Python 3.x
- scikit-learn — MLPClassifier, RandomizedSearchCV
- NumPy — manejo del dataset
- Pandas — tablas de resultados
- Matplotlib — curvas de perdida, matriz de confusion, visualizacion de rostros
