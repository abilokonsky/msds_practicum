import gi
import cv2
import numpy as np
import tensorflow as tf

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Initialize GStreamer
Gst.init(None)

# Load the TensorFlow model
MODEL_DIR = '/home/andrey/msds_practicum/msds_practicum/src/models/trained_models/Model_0002_CNN_ResNet__loss_sparse_categorical_crossentropy'
model = tf.keras.models.load_model(MODEL_DIR)

class_names = [f"{5 * i} degrees" for i in range(1, 20)]

def preprocess_image(image):
    return image.astype('float32') / 255.0

def perform_prediction(image_np):
    prediction = model.predict(np.expand_dims(preprocess_image(image_np), axis=0))
    top_indices = prediction.flatten().argsort()[-3:][::-1]
    top_classes = [class_names[i] for i in top_indices]
    top_probabilities = prediction.flatten()[top_indices]
    return list(zip(top_classes, top_probabilities))

# Define a simple pipeline
pipeline = Gst.parse_launch("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12 ! nvvidconv ! videoconvert ! appsink name=sink emit-signals=True")
appsink = pipeline.get_by_name("sink")

# Start the pipeline
pipeline.set_state(Gst.State.PLAYING)

def capture_and_process():
    sample = appsink.emit("pull-sample")
    if sample:
        buffer = sample.get_buffer()
        caps_format = sample.get_caps().get_structure(0)
        width = caps_format.get_value('width')
        height = caps_format.get_value('height')
        
        # Extract frame data
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return None
        frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape(height, width, 3)
        buffer.unmap(map_info)

        # Process the frame
        predictions = perform_prediction(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        print("Predictions:", predictions)
        return frame  # Optionally process/display the frame elsewhere

try:
    while True:
        frame = capture_and_process()
        if frame is None:
            break
        # Additional frame processing/display can be done here
except KeyboardInterrupt:
    print('Interrupted by user')

finally:
    pipeline.set_state(Gst.State.NULL)
