## Planteamiento del problema a resolver

Planteamiento del problema a resolver  
Durante este parcial, nos enfocamos en la identificación y clasificación de objetos utilizando distintos métodos de visión por computadora, destacando el uso de descriptores. Este trabajo explora la identificación en tiempo real de tres objetos específicos: un avión, un helicóptero y una excavadora, utilizando métodos como BRISK, SURF y HOG para evaluar su eficacia.

## Descripción de la solución

Una vez seleccionados nuestros objetos, empezamos con el la identificacion de estos. En lo personal quize iniciar con BRISK, ya que es lo bastante robusto y rapido para ejecutarse en tiempo real, perfecto para iniciar con el ejercicio.

### BRISK

Lo primero sera importar las librearias que necesitaremos y inicializar el descriptor:

```
from collections import Counter
import cv2
import os

brisk = cv2.BRISK_create()

dataset_dir = './datasets/'  
object_descriptors = {}  
object_keypoints = {}
```

Despues de esto, cree la funcion `process_image` la cual no tiene gran ciencia, a esta se le da una direccion de alguna imagen y la procesa; la lee, la convierte en escala de grises, sin embargo si contiene una linea importante:

```
keypoints, descriptors = brisk.detectAndCompute(gray_image, None)
```

Esta funcion `detectAndCompute` se le otorga una imagen, y regresa los keypoints de esta y ademas el descriptor de la misma. Ahora, lo que creo es la funcion mas importante es `extract_descriptors_from_dataset()`, dentro de esta se crean las direcciones utilizando la libreria `os`, ademas se crean las listas de de descriptores y de keypoints de cada clase, una vez hecho esto, se añadiran a la lista general de descriptores:

```
object_descriptors[object_name] = descriptors_list
object_keypoints[object_name] = keypoints_list
```

Por ultimo, creamos la funcion `compare_descriptors(descriptors_live)` la cual comparara los descriptores de cada frame del video en vivo con los descriptores guardados. Lo primero es crear el matcher bruteForce, esto se hace con la funcion `bf=cv2.BFMatcher()` una vez teniendo este simplemente se comparara y se devolvera el mejor match.

Ademas y para cerrar con brisk, quisiera agregar una pequeña logica que implemente para mi captura en vivo: Note que clasificaba correctamente la mayoria de las veces los frames, sin embargo habia algunos falsos positivos, dados por sombras, diferentes ilumicaciones, etc, por lo que para evitar la impresion volatil del tipo de objeto en pantalla, decidi crear una lista de los objetos detectados en los ultimos 150 frames, y en base a estos mostrar el objeto mas repetido, que segun la logica, deberia ser el objeto en pantalla.

```
concurrente = []
frame_count = 0
 .
 .
 . 
if frame_count > 15:
            obj = contador_obj.most_common(1)[0][0] if contador_obj.most_common(1)[0][0] is not None else 'None'
            cv2.putText(frame, f"Objeto: {obj}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
if frame_count == 150:
            frame_count = 0
            concurrente.clear()
        else:
            frame_count += 1

```

### HOG

Comenzamos importando las bibliotecas necesarias y configurando los parámetros para el descriptor HOG, que extrae características basadas en gradientes de imagen:

```
import cv2
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

hog = cv2.HOGDescriptor((128, 128), (16, 16), (8, 8), (8, 8), 9)
```

Creamos una función de preprocesamiento (`preprocess_image(image)`) que convierte las imágenes a escala de grises y ajusta su contraste, lo que ayuda a mejorar la calidad de los descriptores HOG.

Luego, leemos las imágenes del dataset, un punto que quisiera mencionar es que le tome cerca de 100 fotos a cada objeto, esto con el fin de tener un dataset variado para las pruebas y el test, las preprocesamos y calculamos los descriptores HOG. Estos se almacenan junto con las etiquetas de las clases (avión, excavadora, helicóptero):

```
for label, object_class in enumerate(object_classes):
    for image_name in os.listdir(os.path.join(dataset_path, object_class)):
        image = preprocess_image(cv2.imread(os.path.join(dataset_path, image_name)))
        hog_descriptor = hog.compute(cv2.resize(image, (128, 128)))
        descriptors.append(hog_descriptor.flatten())
        labels.append(label)
```

Dividimos los datos para entrenamiento y prueba, y utilizamos un clasificador Random Forest optimizado con GridSearchCV para encontrar los mejores parámetros:

```
param_grid = {
    'randomforestclassifier__n_estimators': [270, 280],
    'randomforestclassifier__max_depth': [4, 7],
    'randomforestclassifier__min_samples_split': [2, 5],
    'randomforestclassifier__min_samples_leaf': [2, 4]
}

grid = GridSearchCV(pipeline, param_grid, refit=True, cv=5)
grid.fit(X_train, y_train)
```

Por ultimo, e igual que el paso anterior, hacemos una captura de frames en vivo para la clasificacion de los objetos en tiempo real, la unica diferencia es que aqui no acumulo los objetos ni muestro el mayor de 150, esto lo hago para ver explicitamente que se detecta en cada frame, el porque lo explicare en la seccion de resultados.

### SURF -> ORB

Durante el proceso de este trabajo, me di cuenta que SURF ya no esta disponible, al menos no de forma gratuita, se requieren ciertas licencias. Intente instalar una version anterior de openCV y python para tratar de usar SURF sin embargo esto solo causo problemas en mi sistema operativo, para no causar mas desastre decidi usar ORB en lugar de SURF, cabe mencionar que antes jamas lo habia usado por lo que creo que es un buen remplazo para SURF, sin embargo estoy conciente de que no es lo que se nos esta pidiendo y aceptare cualquier penalizacion en la calificacion por esto.

Entrando ahora si al tema de la explicacion, Lo primero será importar las librerías que necesitaremos y inicializar el descriptor:

```
from collections import Counter
import cv2
import os

orb = cv2.ORB_create()

dataset_dir = './datasets/'  
object_descriptors = {}  
object_keypoints = {}
```

Realmente el proceso y la logica es casi identico a BRISK, salvo tal vez por un pequeño detalle que es la linea `orb = cv2.ORB_create()` ahora utilizamos orb en vez de brisk, y esto se repite para las lineas mas importantes:

```
keypoints, descriptors = orb.detectAndCompute(gray_image, None)
```

Como dije, fuera de eso el codigo es practicamente el mismo que BRISK, si se requiere revisar a fondo el archivo esta dentro de `detectorDeObjetos/ORB`

## Descripción de los resultados

### BRISK

La clasificacion en vivo de los objetos resulto ser un exito, sin embargo hubo algunos puntos que creo son necesarios considerar:

- Los objetos se clasifican correctamente siempre y cuando haya una buena iluminacion.
- Ademas, aunque la clasificacion se da en casos contrarios, es recomendable tener los objetos en un fondo blanco (aunque esto en la vida cotidiana no sea una situacion realista) para mayor rapidez y eficacia, se trabajara para mejorar la identificacion con fondos diversos.

Por ultimo agregue un video en tiempo real, este se encuentra dentro de `detectorDeObjetos/Evidencias/BRISK`

### HOG

A diferencia de BRISK, HOG no me dio los resultados que esperaba; aunque en su gran mayoria es notorio que clasifica de manera exitosa, este tiene algunos errores, en especial con la clase excavadora, que no pudo reconocer el objeto en ninguna ocasion, esto tambien se ve reflejado en la presición del modelo:

![aba7d3a3c91fdd978194c3988891d0fb.png](:/92aa5580ce6a4fbf8714c2085e98df45)

Sin embargo y como mencione, clasifica correctamente las clases avion y helicoptero, aunque con unos falsos positivos de por medio, pero se llega a distinguir la dominancia del objeto en pantalla. El video se encuentra en la ruta: `detectorDeObjetos/Evidencias/HOG`

### ORB

Debido a las dificultades para usar SURF use ORB como lo explique antes, entrando al tema de los resultados, siendo este muy similar a BRISK (o al menos en mi opinion) tuvo resultados similares, sin embargo si hay cosas que note.

- ORB genero mas errores que brisk, aunque tambien utilice la logica de mostrar el objeto mas repetido en una lista de frames, tuve que subir el rango a 200 para realizar el video de prueba (aunque para funcion del trabajo, lo volvi a dejar en 150).
- ORB tardaba algunos milisegundos mas que BRISK en detectar los objetos, aunque pueda sonar insignificante creo que en aplicaciones para deteccion de objetos en tiempo real el tiempo es de vital importancia.

El video del detector se puede encontrar en `detectorDeObjetos/Evidencias/ORB`.

### Comparación de descriptores

| Clase | BRISK | HOG | ORB |
| --- | --- | --- | --- |
| Avión | Detectado fácilmente | Detectado fácilmente | Detectado fácilmente |
| Helicóptero | Detectado rápidamente, aunque más lento que el avión | Detectado, pero tardó más que en BRISK | Detectado con ligera demora |
| Excavadora | Detectado con ligera demora | NO DETECTADO | Detectado fácilmente |

## Discusión

Con base en los resultados obtenidos al clasificar mis objetos (avión, helicóptero y excavadora), considero que el descriptor BRISK mostró el mejor desempeño. No solo logró una clasificación precisa en todas las pruebas realizadas, sino que también destacó por su eficiencia, utilizando menos imágenes de referencia en comparación con HOG. Además, su rapidez y flexibilidad al trabajar con una amplia variedad de objetos, gracias a su lógica basada en keypoints, lo hacen una opción muy robusta para tareas de detección y clasificación en tiempo real.

## Conclusión

La clasificación y detección de objetos, ya sea en imágenes estáticas o en videos en tiempo real, es crucial para el avance de muchas tecnologías actuales. Contar con herramientas como OpenCV facilita enormemente el desarrollo en este campo, permitiéndonos aprovechar potentes descriptores sin tener que implementarlos desde cero. Esto nos libera para concentrarnos en otras áreas del proyecto y contribuye a acelerar el progreso en aplicaciones de visión por computadora.
