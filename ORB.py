from collections import Counter
import cv2
import os

# Inicializar ORB (alternativa a SURF)
orb = cv2.ORB_create()

dataset_dir = './datasets/'
object_descriptors = {}
object_keypoints = {}


def process_image(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray_image, None)
        return descriptors, keypoints
    else:
        print(f"No se pudo cargar la imagen {image_path}")
        return None, None


def extract_descriptors_from_dataset():
    for object_name in os.listdir(dataset_dir):
        object_path = os.path.join(dataset_dir, object_name)
        descriptors_list = []
        keypoints_list = []

        if os.path.isdir(object_path):
            for image_name in os.listdir(object_path):
                if image_name.endswith(('.jpg', '.png')):
                    image_path = os.path.join(object_path, image_name)
                    descriptors, keypoints = process_image(image_path)
                    if descriptors is not None:
                        descriptors_list.append(descriptors)
                        keypoints_list.append(keypoints)

            object_descriptors[object_name] = descriptors_list
            object_keypoints[object_name] = keypoints_list
        else:
            print(f"Advertencia: {object_path} no es un directorio.")


extract_descriptors_from_dataset()


def compare_descriptors(descriptors_live):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    best_match = None
    best_score = 0
    MIN_GOOD_MATCHES = 7
    DISTANCE_THRESHOLD = 75

    for object_name, descriptors_list in object_descriptors.items():
        for descriptors in descriptors_list:
            matches = bf.match(descriptors, descriptors_live)
            matches = sorted(matches, key=lambda x: x.distance)

            good_matches = [m for m in matches if m.distance < DISTANCE_THRESHOLD]
            score = len(good_matches)

            if score > best_score and score >= MIN_GOOD_MATCHES:
                best_score = score
                best_match = object_name

    return best_match, best_score


# Captura de video en tiempo real
cap = cv2.VideoCapture(0)
concurrente = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar el frame")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_keypoints, frame_descriptors = orb.detectAndCompute(gray_frame, None)

    if frame_descriptors is not None:
        best_match_index, match_count = compare_descriptors(frame_descriptors)
        concurrente.append(best_match_index)
        contador_obj = Counter(concurrente)

        if frame_count > 15:
            obj = contador_obj.most_common(1)[0][0] if contador_obj.most_common(1)[0][0] is not None else 'None'
            cv2.putText(frame, f"Objeto: {obj}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Clasificación en Tiempo Real", frame)

        if frame_count == 150:
            frame_count = 0
            concurrente.clear()
        else:
            frame_count += 1
    else:
        cv2.imshow("Clasificación en Tiempo Real", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
