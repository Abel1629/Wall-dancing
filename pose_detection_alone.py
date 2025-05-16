# ---------------------- Imports ---------------------- #
import os
import sys
import cv2
import numpy as np
import joblib

import socket
from threading import Thread

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# ---------------------- Global Settings ---------------------- #

HOST = '127.0.0.1'  # Localhost
PORT = 65432        # Port to send data to

USE_UNITY = False

# ---------------------- Global placeholders ---------------------- #
interpreter = input_details = output_details = None
scaler = None
pose = None
cap = None

# ---------------------- Functions ---------------------- #

def init_pose_model():
    # set up Mediapipe
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    return mp_pose.Pose(static_image_mode=True)

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

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
    
def detect_pose_tflite(photo, interpreter, input_details, output_details, scaler, pose):
    keypoints_array = extract_keypoints(photo, pose)
    if len(keypoints_array) == 33 * 4:
        X_new = scaler.transform([keypoints_array]).astype(np.float32)
        
        # Set input and invoke interpreter
        interpreter.set_tensor(input_details[0]['index'], X_new)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        confidence = np.max(output_data)
        if confidence < 0.6:
            return 404
        else:
            predicted_pose = np.argmax(output_data)
            return predicted_pose
    else:
        return 404 # no pose detected because keypoints where not extracted correctly


def send_message(message, client):
    try:
        msg = f"{message}\n"
        client.sendall(msg.encode('utf-8'))
    except Exception as e:
        print("Error sending message:", e)

def resource_path(relative_path):
    try:
        # When using PyInstaller
        base_path = sys._MEIPASS
    except AttributeError:
        # When running normally
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ---------------------- Load Everything in Background ---------------------- #
def preload_assets():
    global interpreter, input_details, output_details, scaler, pose, cap

    save_dir = resource_path('TrainedModels')

    # Load model & scaler
    scaler = joblib.load(os.path.join(save_dir, 'scaler.pkl'))
    # mlp_model = tf.keras.models.load_model(os.path.join(save_dir, 'pose_classifier_model.h5'))
    interpreter, input_details, output_details = load_tflite_model(os.path.join(save_dir, 'pose_classifier_model.tflite'))

    # Init pose model
    pose = init_pose_model()

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)


# ---------------------- Main Script ---------------------- #
if __name__ == "__main__":
    # Start loading in a separate thread while waiting for Unity
    loader_thread = Thread(target=preload_assets)
    loader_thread.start()

    if USE_UNITY:
        # Wait for Unity connection
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((HOST, PORT))
        server.listen(1)
        print("Waiting for Unity connection...")
        client, addr = server.accept()
        print("Connected to Unity at", addr)

    # Wait until model/camera are ready
    loader_thread.join()

    if USE_UNITY:
        # Let Unity know Python is ready
        send_message("READY", client)

    while True:
        success, frame = cap.read()
        if not success:
            break
        mirrored_frame = cv2.flip(frame, 1) # mirrow image as mlp was trained with camera images
        cv2.imshow('frame', mirrored_frame)
        # pose_num = detect_pose(frame, mlp_model, scaler, pose)
        pose_num = detect_pose_tflite(frame, interpreter, input_details, output_details, scaler, pose)
        print(f"Detected pose: {pose_num}")

        if (pose_num <= 12 and pose_num >= 0) and USE_UNITY:
            send_message(pose_num, client)

        if cv2.waitKey(5) & 0xFF == ord('q'):  # Q to quit
            break

    cap.release()
    if USE_UNITY:
        client.close()
        server.close()
    cv2.destroyAllWindows()