import cv2
import mediapipe as mp
import math
import webbrowser
import time

cap = cv2.VideoCapture(0)

hands = mp.solutions.hands.Hands(static_image_mode=False,
                         max_num_hands=1,
                         min_tracking_confidence=0.5,
                         min_detection_confidence=0.5)

mpDraw = mp.solutions.drawing_utils

while True:
    _, img = cap.read()

    result = hands.process(img)

    if result.multi_hand_landmarks:
        for id, lm in enumerate(result.multi_hand_landmarks[0].landmark):
            if id == 4:
                four_coordinates = lm
            if id == 12:
                eight_coordinates = lm

        if math.fabs(eight_coordinates.x - four_coordinates.x) > 0.007 and \
                math.fabs(eight_coordinates.y - four_coordinates.y) > 0.007 and \
                math.fabs(eight_coordinates.z - four_coordinates.z) > 0.007:
                webbrowser.open_new_tab('https://google.com')
                time.sleep(5)


        for i in range(len(result.multi_hand_landmarks)):
            mpDraw.draw_landmarks(img, result.multi_hand_landmarks[i], mp.solutions.hands.HAND_CONNECTIONS)


    cv2.imshow('Hand tracking', img)
    cv2.waitKey(1)