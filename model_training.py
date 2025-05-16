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

import tensorflow as tf
from tensorflow.keras import layers, models

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# ---------------------- Global Settings ---------------------- #
mode_colab = False

visualize = True

if mode_colab:
    from google.colab import drive
    drive.mount('/content/drive')
    work_path = "/content/drive/My Drive/Scbio Project"
else:
    work_path = os.getcwd()

# ---------------------- Mediapipe Setup ---------------------- #
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

# ---------------------- Functions ---------------------- #

def generate_dataframe(data_path, pose):
    failed_count = 0
    keypoint_columns = [f"{joint}_{coord}" for joint in range(33) for coord in ['x', 'y', 'z', 'vis']]
    data = []

    for pose_folder in os.listdir(data_path):
        pose_path = os.path.join(data_path, pose_folder)
        label = int(pose_folder.split('_')[0])

        for sample_image in os.listdir(pose_path):
            image_path = os.path.join(pose_path, sample_image)
            image = cv2.imread(image_path)

            if image is None:
                failed_count += 1
                continue

            keypoints = extract_keypoints(image, pose)
            if len(keypoints) != 33 * 4:
                failed_count += 1
                continue

            row = {
                'image': image,
                'label': label,
                **dict(zip(keypoint_columns, keypoints))
            }

            data.append(row)
        print(f'Images from class {label} added to dataframe')


    df_poses = pd.DataFrame(data)
    return df_poses, failed_count, keypoint_columns


def extract_keypoints(image, pose):
    results = pose.process(image)
    keypoints = []

    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])

    return keypoints


def visualize_keypoints_per_class(df_poses, pose_num_map, keypoint_columns, mp_pose):
    mp_drawing = mp.solutions.drawing_utils
    POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

    num_classes = len(df_poses['label'].unique())
    # set up grid for plot
    cols = 4
    rows = math.ceil(num_classes / cols)
    plt.figure(figsize=(10,8))

    for pose_num in df_poses['label'].unique():
        pose_indices = df_poses[df_poses['label'] == pose_num].index
        random_index = random.choice(pose_indices)
        sample = df_poses.loc[random_index]

        keypoints = sample[keypoint_columns].to_numpy(dtype=np.float32)
        image = df_poses['image'][random_index]

        landmarks = []
        for i in range(33):
            lm = landmark_pb2.NormalizedLandmark()
            lm.x = keypoints[i * 4 + 0]
            lm.y = keypoints[i * 4 + 1]
            lm.z = keypoints[i * 4 + 2]
            lm.visibility = keypoints[i * 4 + 3]
            landmarks.append(lm)

        landmark_list = landmark_pb2.NormalizedLandmarkList(landmark=landmarks)
        mp_drawing.draw_landmarks(image, landmark_list, POSE_CONNECTIONS)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plt.subplot(rows, cols, pose_num+1)
        plt.imshow(image_rgb)
        plt.title(f"Pose: {pose_num_map[pose_num]}")
        plt.axis('off')

    plt.savefig('keypoints_per_class.png')


def build_pose_classifier(input_dim=132, num_classes=13):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def preprocess_data(df_poses, keypoint_columns, save_dir):
    X = df_poses[keypoint_columns].values.astype('float32')
    y = df_poses['label'].values.astype('int32')

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    return X_train, X_test, y_train, y_test


def train_save_mlp(model, X_train, y_train, X_val, y_val, save_dir, epochs=100, batch_size=32):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    )
    model_path = os.path.join(save_dir, 'pose_classifier_model.h5')
    model.save(model_path)

    model_path_keras = os.path.join(save_dir, "pose_classifier_model.keras")
    model.save(model_path_keras)
    return history


def visualize_training_history(history):
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    plt.figure(figsize=(8,8))
    plt.plot(epochs, history.history['accuracy'])
    plt.plot(epochs, history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.savefig('training_accuracy.png')

    plt.figure(figsize=(8,8))
    plt.plot(epochs, history.history['loss'], label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.savefig('training_loss.png')


def predict_and_visualize(X_val, y_val, model, pose_num_map):
    y_pred_probs = model.predict(X_val)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_val, y_pred)
    labels = [pose_num_map[i] for i in range(len(pose_num_map))]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    plt.savefig('confusion_matrix.png')

    return y_pred


# ---------------------- Main Script ---------------------- #
if __name__ == "__main__":
    data_path = os.path.join(work_path, 'Photos_Scbio')
    save_dir = os.path.join(work_path, 'TrainedModels')
    os.makedirs(save_dir, exist_ok=True)  # Creates the folder if it doesn't exist

    pose_num_map = {
        0: 'Base',
        1: 'T',
        2: 'Star',
        3: 'Crouch',
        4: 'Tree Right Up',
        5: 'Tree Right Down',
        6: 'Tree Left Up',
        7: 'Tree Left Down',
        8: 'Y',
        9: 'Leg up right',
        10: 'Leg up left',
        11: 'Hands head',
        12: 'Surfer',
    }

    # Generate structure containing training images, class labels and all 33 keypoints
    df_poses, failed_count, keypoint_columns = generate_dataframe(data_path, pose)

    print('Shape of the dataframe: ', df_poses.shape)
    print(df_poses.value_counts('label'))

    

    # --- Built and train MLP for pose classification ---
    model = build_pose_classifier()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    X_train, X_test, y_train, y_test = preprocess_data(df_poses, keypoint_columns, save_dir)
    history = train_save_mlp(model, X_train, y_train, X_test, y_test, save_dir)
    
    if visualize:
        # Visualize keypoints
        visualize_keypoints_per_class(df_poses, pose_num_map, keypoint_columns, mp_pose)
        visualize_training_history(history)

        y_pred = predict_and_visualize(X_test, y_test, model, pose_num_map)

        plt.show()