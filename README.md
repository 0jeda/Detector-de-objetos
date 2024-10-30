Identificación y Clasificación de Objetos en Tiempo Real con Descriptores de Imagen
Planteamiento del Problema
Este proyecto se centra en la identificación y clasificación de objetos en tiempo real utilizando métodos de visión por computadora y descriptores de imagen. Se han evaluado tres objetos específicos (avión, helicóptero y excavadora) usando los descriptores BRISK, ORB y HOG para medir su eficacia en distintas condiciones.

Descripción de la Solución
Selección de Objetos y Descriptores
Para la identificación de los objetos, se comenzó con el descriptor BRISK, debido a su robustez y rapidez para ejecutar tareas en tiempo real. Posteriormente, se implementaron HOG y ORB para hacer comparaciones y observar el comportamiento de cada descriptor bajo las mismas condiciones.

BRISK
Se inició el proceso importando las bibliotecas necesarias y configurando el descriptor:

python
Copiar código
from collections import Counter
import cv2
import os

brisk = cv2.BRISK_create()
dataset_dir = './datasets/'
object_descriptors = {}
object_keypoints = {}
La función process_image() procesa cada imagen convirtiéndola a escala de grises y utilizando detectAndCompute para obtener keypoints y descriptores. La función principal extract_descriptors_from_dataset() organiza estos datos por clase, almacenándolos para su uso en la identificación en tiempo real.

Para minimizar falsos positivos, se implementó una lista que acumula los objetos detectados en los últimos 150 frames, mostrando el objeto más frecuente:

python
Copiar código
concurrente = []
frame_count = 0
# ... lógica de acumulación ...
if frame_count > 15:
    obj = contador_obj.most_common(1)[0][0] if contador_obj.most_common(1)[0][0] is not None else 'None'
    cv2.putText(frame, f"Objeto: {obj}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
HOG
Se configuraron parámetros para el descriptor HOG para extraer características basadas en gradientes de imagen. Luego se creó una función de preprocesamiento que convierte las imágenes a escala de grises y mejora su contraste. Para el dataset, se tomaron aproximadamente 100 fotos de cada objeto para garantizar variedad en las pruebas. Se entrenó un clasificador Random Forest optimizado mediante GridSearchCV.

python
Copiar código
param_grid = {
    'randomforestclassifier__n_estimators': [270, 280],
    'randomforestclassifier__max_depth': [4, 7],
    'randomforestclassifier__min_samples_split': [2, 5],
    'randomforestclassifier__min_samples_leaf': [2, 4]
}

grid = GridSearchCV(pipeline, param_grid, refit=True, cv=5)
grid.fit(X_train, y_train)
ORB (en sustitución de SURF)
Dado que SURF no está disponible de forma gratuita, se optó por ORB. El proceso y lógica son similares a los de BRISK, con el ajuste de utilizar cv2.ORB_create() para crear el descriptor.

python
Copiar código
orb = cv2.ORB_create()
Descripción de Resultados
BRISK
BRISK mostró un alto rendimiento en la clasificación en tiempo real, aunque los resultados dependen de una buena iluminación. Para mejorar la eficacia, es preferible que los objetos estén en un fondo blanco.

HOG
HOG funcionó bien para los objetos avión y helicóptero, pero tuvo problemas en la detección de la excavadora. Los resultados se guardan en detectorDeObjetos/Evidencias/HOG.

ORB
ORB fue efectivo, aunque mostró más errores en comparación con BRISK. Se ajustó el conteo de frames a 200 para estabilizar la detección en los videos de prueba.

Comparación de Desempeño
Objeto	BRISK	HOG	ORB
Avión	Detectado fácilmente	Detectado fácilmente	Detectado fácilmente
Helicóptero	Detectado rápidamente	Detectado con ligera demora	Detectado con ligera demora
Excavadora	Detectado con ligera demora	No detectado	Detectado fácilmente
Discusión
Con base en los resultados, BRISK demostró ser el descriptor más eficaz en términos de precisión y velocidad. Su habilidad para trabajar con menos imágenes de referencia hace que sea adecuado para aplicaciones en tiempo real.

Conclusión
La clasificación y detección de objetos es fundamental para aplicaciones actuales de visión por computadora. OpenCV facilita la implementación de estos algoritmos, permitiendo que el enfoque esté en la aplicación del proyecto y acelerando el desarrollo en el campo.
