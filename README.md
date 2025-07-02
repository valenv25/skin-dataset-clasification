# Ejemplo de clasificación de imágenes con PyTorch y MLP (+CNN)

### skin-dataset-classification

El modelo base presentado se encuentra en src/base, y fue utilizado para obtener una primera aproximación a la solución buscada: entrenar un modelo (simple, en principio) de redes neuronales para clasificar imágenes. El desarrollo principal del modelo avanzado que incluye las modificaciones propuestas se fue construyendo en `Clasificacion.ipynb`, pero se realizaron pruebas con Keras y PyTorch (MLP) en `Clasificacion_Keras.ipynb` y `Clasificacion_PyTorch_MLP.ipynb` para testear las APIs y frameworks (usan un dataset alternativo). `Clasificacion_PyTorch_MLP_Avanzada.ipynb` es un mix intermedio que contiene algunos cambios para probar `Albumentations` y un par de adiciones simples como `weight_decay`, pero usando un `train` y un `evaluate` más simples y compactos, sin MLflow.

No se realiza búsqueda de hiperparámetros por el tiempo excesivo que lleva recorrer los loops anidados. Las actividades de modificación se encuentran detalladas en `Mods.ipynb`, pero para realizar las pruebas se debe usar el modelo `MLPClassifierComplete()` que incluye las variantes testeadas. Para ir respondiendo las preguntas teóricas y (especialmente) las prácticas se fueron variando los parámetros de cada modelo para luego visualizarlos y compararlos en MLflow y Tensorboard (en los puertos 5000 y 6006), pero los runs del experimento y los modelos guardados utilizan solo un caso para demostración.

Adicionalmente, se intercambia el clasificador MLP por una CNN (Convolutional Neural Networks, ConvNets) para ver cómo varían los resultados y analizar los cambios. La experimentación con el nuevo modelo se realiza en `Clasificacion_CNN.ipynb` (y se renombró `Clasificacion.ipynb` a `Clasificacion_MLP.ipynb`). Algunos comentarios sobre las modificaciones se hicieron en `Adicional_CNN.md`.

---

Para generar el entorno

```
conda create -n {env}  
conda activate {env}  
conda install -c conda-forge python==3.12.3  
pip install -r requirements.txt  
```

---
