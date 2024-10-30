import cv2
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


win_size = (128, 128)
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
nbins = 9

hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

# Ruta del dataset
dataset_path = './dataset_HOG/'
object_classes = ['avion', 'excavadora', 'helicoptero']


descriptors = []
labels = []



def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.equalizeHist(gray_image)
    return gray_image

for label, object_class in enumerate(object_classes):
    object_path = os.path.join(dataset_path, object_class)
    for image_name in os.listdir(object_path):
        image_path = os.path.join(object_path, image_name)
        print(image_path)
        image = cv2.imread(image_path)
        if image is not None:
            # Preprocesamos la imagen
            image = preprocess_image(image)
            # Redimensionamos las imágenes
            image = cv2.resize(image, (128, 128))
            # Calculamos el descriptor HOG
            hog_descriptor = hog.compute(image)
            descriptors.append(hog_descriptor.flatten())
            labels.append(label)

descriptors = np.array(descriptors)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(descriptors, labels, test_size=0.2, random_state=42)

pipeline = make_pipeline(StandardScaler(), RandomForestClassifier())

param_grid = {
    'randomforestclassifier__n_estimators': [270, 280],
    'randomforestclassifier__max_depth': [4, 7],
    'randomforestclassifier__min_samples_split': [2, 5],
    'randomforestclassifier__min_samples_leaf': [2, 4]
}

grid = GridSearchCV(pipeline, param_grid, refit=True, cv=5)
grid.fit(X_train, y_train)

accuracy = grid.score(X_test, y_test)
print(f"Precisión del clasificador con Random Forest: {accuracy * 100:.2f}%")
print(f"Mejores parámetros encontrados: {grid.best_params_}")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = preprocess_image(frame)  # Preprocesamos el frame
    gray_frame = cv2.resize(gray_frame, (128, 128))
    hog_descriptor = hog.compute(gray_frame).flatten()
    predicted_class = grid.predict([hog_descriptor])[0]

    object_name = object_classes[predicted_class]
    cv2.putText(frame, object_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Detección HOG en Tiempo Real', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
