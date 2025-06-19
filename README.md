# Ejemplo de clasificación de imágenes con PyTorch y MLP

### skin-dataset-classification

El modelo base presentado se encuentra en src/base, y fue utilizado para obtener una primera aproximación a la solución buscada: entrenar un modelo (simple, en principio) de redes neuronales para clasificar imágenes. El desarrollo principal del modelo avanzado que incluye las modificaciones propuestas se fue construyendo en `Clasificacion.ipynb`, pero se realizaron pruebas con Keras y PyTorch (MLP) en `Clasificacion_Keras.ipynb` y `Clasificacion_PyTorch_MLP.ipynb` para testear las APIs y frameworks (usan un dataset alternativo). `Clasificacion_PyTorch_MLP_Avanzada.ipynb` es un mix intermedio que contiene algunos cambios para probar `Albumentations` y un par de adiciones simples como `weight_decay`, pero usando un `train` y un `evaluate` mas simple y compacto, sin MLflow. No se realiza búsqueda de hiperparámetros por el tiempo escesivo que lleva recorrer los loops anidados.

---

Para generar el entorno

```
conda create -n {env}  
conda activate {env}  
conda install -c conda-forge python==3.12.3  
pip install -r requirements.txt  
```

---