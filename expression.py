import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

try:
    imagem_neutra = cv2.imread("neutro.jpg")
    imagem_sorrindo = cv2.imread("sorrindo.jpg")
    imagem_thumbs_up = cv2.imread("thumbs_up.jpg")

    imagem_neutra = cv2.resize(imagem_neutra, (400, 400))
    imagem_sorrindo = cv2.resize(imagem_sorrindo, (400, 400))
    imagem_thumbs_up = cv2.resize(imagem_thumbs_up, (400, 400))

except Exception as e:
    print(f"Verifique se as imagens estao na pasta{e}")
    exit()

def is_thumb_up(hand_landmarks):

    try:
        thumb_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
        thumb_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y

        index_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        index_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y

        middle_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
        middle_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y

        if thumb_tip_y < thumb_mcp_y and index_tip_y > index_pip_y and middle_tip_y > middle_pip_y:
            return True
    except:
        return False
    return False

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("nao foi possivel abrir a camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results_hands = hands.process(image_rgb)
    
    estado_atual = 'nenhum'

    if results_hands.multi_hand_landmarks and len(results_hands.multi_hand_landmarks) == 2:

        hand1_is_up = is_thumb_up(results_hands.multi_hand_landmarks[0])
        hand2_is_up = is_thumb_up(results_hands.multi_hand_landmarks[1])
        
        if hand1_is_up and hand2_is_up:
            estado_atual = 'thumbs_up'

    if estado_atual != 'thumbs_up':
        try:
            resultado_face = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            
            if isinstance(resultado_face, list) and len(resultado_face) > 0:
                emocao_dominante = resultado_face[0]['dominant_emotion']

                if emocao_dominante == 'sad':
                    emocao_dominante = 'neutral'
                
                if emocao_dominante == 'happy':
                    estado_atual = 'happy'
                elif emocao_dominante == 'neutral':
                    estado_atual = 'neutral'
        except Exception as e:
            pass

    if estado_atual == 'thumbs_up':
        cv2.imshow('Expressao', imagem_thumbs_up)
    elif estado_atual == 'happy':
        cv2.imshow('Expressao', imagem_sorrindo)
    elif estado_atual == 'neutral':
        cv2.imshow('Expressao', imagem_neutra)
    else:
        imagem_padrao = np.zeros((400, 400, 3), dtype="uint8")
        cv2.imshow('Expressao', imagem_padrao)

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
