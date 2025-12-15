from model.MobileNetV2 import MobileNetV2
from data.data_infos import get_classes, get_dimension
from data.data_treatment import image_treatment
import numpy as np
import streamlit as st
import cv2
import os
import mediapipe as mp

# Chemin relatif vers le modèle
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model', 'static', 'MobileNetV2_model_224x224_2.h5')

model = MobileNetV2(model_path)
model.load()
classes = get_classes()
dimension = get_dimension()


def predict_sign(image):
    img = image_treatment(image, dimension)
    predictions = model.predict(img)
    # st.image(img[0], channels='RGB', caption='Image Prétraitée')
    class_idx = np.argmax(predictions)
    sign = classes[class_idx]
    probability = round(predictions[0][class_idx]*100, 2)  # Obtenir la probabilité de la classe prédite

    return sign, probability


def stream_window():
    # Interface Streamlit
    st.title('Reconnaissance de Signes en Temps Réel')
    run = st.checkbox('Run', key='run_checkbox')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    return run, camera, FRAME_WINDOW


def detect_and_crop_hand(frame, hands):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:


            # Obtenir les coordonnées de la main
            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]

            x_min = int(min(x_coords) * frame.shape[1])
            x_max = int(max(x_coords) * frame.shape[1])
            y_min = int(min(y_coords) * frame.shape[0])
            y_max = int(max(y_coords) * frame.shape[0])

            # Debugging: Print the coordinates
            print(f"x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}")

            # Ajouter une marge autour de la main
            margin_x = 60
            margin_y = 70
            x_min = max(0, x_min - margin_x)
            x_max = min(frame.shape[1], x_max + margin_x)
            y_min = max(0, y_min - margin_y)
            y_max = min(frame.shape[0], y_max + margin_y)

            # Debugging: Print the coordinates
            print(f"x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}")

            # Dessiner un rectangle autour de la main pour visualiser le recadrage
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            # Assurez-vous que les dimensions sont valides
            if x_max > x_min and y_max > y_min:
                # Recadrer l'image autour de la main
                cropped_frame = frame[y_min:y_max, x_min:x_max]
                return cropped_frame

    return frame


if __name__ == '__main__':
    run, camera, FRAME_WINDOW = stream_window()
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    while run:
        ret, frame = camera.read()
        if not ret:
            st.write("Erreur de capture vidéo")
            break

        cropped_frame = detect_and_crop_hand(frame, hands)

        frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

        # Prédire le signe sur l'image actuelle
        sign, proba = predict_sign(frame_rgb)

        # Afficher la prédiction sur le flux vidéo
        cv2.putText(frame, f'Sign: {sign} / {proba}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Convertir l'image `frame` en RGB pour l'affichage
        frame_rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        FRAME_WINDOW.image(frame_rgb_display)

        cv2.waitKey(1)
    else:
        st.write('Stop')
        camera.release()
        cv2.destroyAllWindows()
        hands.close()
