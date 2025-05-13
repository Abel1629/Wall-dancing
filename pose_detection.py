# ---------------------- Imports ---------------------- #
import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import math

import socket

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# ---------------------- Global Settings ---------------------- #

HOST = '127.0.0.1'  # Localhost
PORT = 65432        # Port to send data to

# ---------------------- Mediapipe Setup ---------------------- #
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

# ---------------------- Functions ---------------------- #

def extract_keypoints(image, pose):
    results = pose.process(image)
    keypoints = []

    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])

    return keypoints

def detect_pose(photo, model, scaler, pose):
    keypoints_array = extract_keypoints(photo, pose)
    if len(keypoints_array) == 33 * 4:
        # Preprocess new data before predicting
        X_new = scaler.transform([keypoints_array])
        predictions = model.predict(X_new)
        confidence = np.max(predictions)
        # safety threshold that no pose is sent if model is very unsure
        if confidence < 0.6:
            return 404
        else:
            predicted_pose = np.argmax(predictions, axis=1).item()
            return predicted_pose
    else:
        return 404 # no pose detected because keypoints where not extracted correctly

def send_pose_number(num):
    try:
        sock.sendall(str(num).encode('utf-8'))
    except Exception as e:
        print("Error sending pose number:", e)


# ---------------------- Main Script ---------------------- #
if __name__ == "__main__":    
    save_dir = os.path.join(os.getcwd(), 'TrainedModels')
    scaler = joblib.load(os.path.join(save_dir, 'scaler.pkl'))
    mlp_model = tf.keras.models.load_model(os.path.join(save_dir, 'pose_classifier_model.h5'))

    # set up connection to game
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))

    # open camera
    cap = cv2.VideoCapture(0)

    # receive variables from game that game is started
    # maybe substitute and start script when game is startet??
    in_game = True

    while (in_game):
        success, frame = cap.read()
        if not success:
            break
        mirrored_frame = cv2.flip(frame, 1) # mirrow image as mlp was trained with camera images
        cv2.imshow('frame', mirrored_frame)
        pose_num = detect_pose(mirrored_frame, mlp_model, scaler, pose)
        print(f"Detected pose: {pose_num}")

        if (pose_num <= 12 and pose_num >= 0):
            send_pose_number(pose_num)

        if cv2.waitKey(5) & 0xFF == ord('q'):  # Q to quit
            break

    cap.release()
    sock.close()
    cv2.destroyAllWindows()