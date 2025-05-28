import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model("TrainedModels\pose_classifier_model.keras")

# Create the TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: Optimization (makes model smaller/faster)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Save to a .tflite file
with open("TrainedModels\pose_classifier_model.tflite", "wb") as f:
    f.write(tflite_model)
