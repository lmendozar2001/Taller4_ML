# Taller 4: Comparación de Modelos Clásicos de ML vs Deep Learning

Implementación y comparación de modelos clásicos de Machine Learning (SVM, Random Forest, KNN, Regresión Logística) con Deep Learning (MLP) para clasificación multiclase del dataset Olivetti Faces.

## Dataset

- **Nombre:** Olivetti Faces
- **Muestras:** 400 imágenes (40 personas, 10 imágenes por persona)
- **Dimensiones:** 64x64 píxeles = 4096 características
- **Clases:** 40 (clasificación multiclase)

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
jupyter notebook Taller4_ML_Clasico.ipynb
```

Ejecuta las celdas en orden. Tiempo estimado: 10-15 minutos.

## Modelos Implementados

1. **SVM** - Support Vector Machine con kernel RBF
2. **Random Forest** - Ensemble de árboles
3. **KNN** - K-Nearest Neighbors
4. **Regresión Logística** - Modelo lineal
5. **MLP** - Multi-Layer Perceptron (Deep Learning)

## Estructura del Notebook

1. Introducción y Preprocesamiento
2. Configuración de Modelos
3. Búsqueda de Hiperparámetros
4. Entrenamiento del MLP
5. Análisis Comparativo
6. Visualizaciones
7. Análisis Técnico
8. Conclusiones

## Resultados Generados

Al ejecutar el notebook se generan:
- Tablas comparativas de resultados
- Gráficos de comparación de modelos
- Matrices de confusión
- Análisis de overfitting
- Modelos entrenados (.pkl)

## Autor

Luis Mendoza - [@lmendozar2001](https://github.com/lmendozar2001)
