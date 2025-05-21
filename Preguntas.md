
# Preguntas sobre el ejemplo de clasificación de imágenes con PyTorch y MLP

## 1. Dataset y Preprocesamiento
- ¿Por qué es necesario redimensionar las imágenes a un tamaño fijo para una MLP?
- ¿Qué ventajas ofrece Albumentations frente a otras librerías de transformación como `torchvision.transforms`?
- ¿Qué hace `A.Normalize()`? ¿Por qué es importante antes de entrenar una red?
- ¿Por qué convertimos las imágenes a `ToTensorV2()` al final de la pipeline?

## 2. Arquitectura del Modelo
- ¿Por qué usamos una red MLP en lugar de una CNN aquí? ¿Qué limitaciones tiene?
- ¿Qué hace la capa `Flatten()` al principio de la red?
- ¿Qué función de activación se usó? ¿Por qué no usamos `Sigmoid` o `Tanh`?
- ¿Qué parámetro del modelo deberíamos cambiar si aumentamos el tamaño de entrada de la imagen?

## 3. Entrenamiento y Optimización
- ¿Qué hace `optimizer.zero_grad()`?
- ¿Por qué usamos `CrossEntropyLoss()` en este caso?
- ¿Cómo afecta la elección del tamaño de batch (`batch_size`) al entrenamiento?
- ¿Qué pasaría si no usamos `model.eval()` durante la validación?

## 4. Validación y Evaluación
- ¿Qué significa una accuracy del 70% en validación pero 90% en entrenamiento?
- ¿Qué otras métricas podrían ser más relevantes que accuracy en un problema real?
- ¿Qué información útil nos da una matriz de confusión que no nos da la accuracy?
- En el reporte de clasificación, ¿qué representan `precision`, `recall` y `f1-score`?

## 5. TensorBoard y Logging
- ¿Qué ventajas tiene usar TensorBoard durante el entrenamiento?
- ¿Qué diferencias hay entre loguear `add_scalar`, `add_image` y `add_text`?
- ¿Por qué es útil guardar visualmente las imágenes de validación en TensorBoard?
- ¿Cómo se puede comparar el desempeño de distintos experimentos en TensorBoard?

## 6. Generalización y Transferencia
- ¿Qué cambios habría que hacer si quisiéramos aplicar este mismo modelo a un dataset con 100 clases?
- ¿Por qué una CNN suele ser más adecuada que una MLP para clasificación de imágenes?
- ¿Qué problema podríamos tener si entrenamos este modelo con muy pocas imágenes por clase?
- ¿Cómo podríamos adaptar este pipeline para imágenes en escala de grises?

## 7. Regularización

### Preguntas teóricas:
- ¿Qué es la regularización en el contexto del entrenamiento de redes neuronales?
- ¿Cuál es la diferencia entre `Dropout` y regularización `L2` (weight decay)?
- ¿Qué es `BatchNorm` y cómo ayuda a estabilizar el entrenamiento?
- ¿Cómo se relaciona `BatchNorm` con la velocidad de convergencia?
- ¿Puede `BatchNorm` actuar como regularizador? ¿Por qué?
- ¿Qué efectos visuales podrías observar en TensorBoard si hay overfitting?
- ¿Cómo ayuda la regularización a mejorar la generalización del modelo?

### Actividades de modificación:
1. Agregar Dropout en la arquitectura MLP:
   - Insertar capas `nn.Dropout(p=0.5)` entre las capas lineales y activaciones.
   - Comparar los resultados con y sin `Dropout`.

2. Agregar Batch Normalization:
   - Insertar `nn.BatchNorm1d(...)` después de cada capa `Linear` y antes de la activación:
     ```python
     self.net = nn.Sequential(
         nn.Flatten(),
         nn.Linear(in_features, 512),
         nn.BatchNorm1d(512),
         nn.ReLU(),
         nn.Dropout(0.5),
         nn.Linear(512, 256),
         nn.BatchNorm1d(256),
         nn.ReLU(),
         nn.Dropout(0.5),
         nn.Linear(256, num_classes)
     )
     ```

3. Aplicar Weight Decay (L2):
   - Modificar el optimizador:
     ```python
     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
     ```

4. Reducir overfitting con data augmentation:
   - Agregar transformaciones en Albumentations como `HorizontalFlip`, `BrightnessContrast`, `ShiftScaleRotate`.

5. Early Stopping (opcional):
   - Implementar un criterio para detener el entrenamiento si la validación no mejora después de N épocas.

### Preguntas prácticas:
- ¿Qué efecto tuvo `BatchNorm` en la estabilidad y velocidad del entrenamiento?
- ¿Cambió la performance de validación al combinar `BatchNorm` con `Dropout`?
- ¿Qué combinación de regularizadores dio mejores resultados en tus pruebas?
- ¿Notaste cambios en la loss de entrenamiento al usar `BatchNorm`?

## 8. Inicialización de Parámetros

### Preguntas teóricas:
- ¿Por qué es importante la inicialización de los pesos en una red neuronal?
- ¿Qué podría ocurrir si todos los pesos se inicializan con el mismo valor?
- ¿Cuál es la diferencia entre las inicializaciones de Xavier (Glorot) y He?
- ¿Por qué en una red con ReLU suele usarse la inicialización de He?
- ¿Qué capas de una red requieren inicialización explícita y cuáles no?

### Actividades de modificación:
1. Agregar inicialización manual en el modelo:
   - En la clase `MLP`, agregar un método `init_weights` que inicialice cada capa:
     ```python
     def init_weights(self):
         for m in self.modules():
             if isinstance(m, nn.Linear):
                 nn.init.kaiming_normal_(m.weight)
                 nn.init.zeros_(m.bias)
     ```

2. Probar distintas estrategias de inicialización:
   - Xavier (`nn.init.xavier_uniform_`)
   - He (`nn.init.kaiming_normal_`)
   - Aleatoria uniforme (`nn.init.uniform_`)
   - Comparar la estabilidad y velocidad del entrenamiento.

3. Visualizar pesos en TensorBoard:
   - Agregar esta línea en la primera época para observar los histogramas:
     ```python
     for name, param in model.named_parameters():
         writer.add_histogram(name, param, epoch)
     ```

### Preguntas prácticas:
- ¿Qué diferencias notaste en la convergencia del modelo según la inicialización?
- ¿Alguna inicialización provocó inestabilidad (pérdida muy alta o NaNs)?
- ¿Qué impacto tiene la inicialización sobre las métricas de validación?
- ¿Por qué `bias` se suele inicializar en cero?