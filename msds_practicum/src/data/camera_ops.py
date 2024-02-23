import gi

import cv2
import numpy as np
import tensorflow as tf
from gi.repository import Gst, GLib


# Initialize GStreamer
Gst.init(None)

# Load the TensorFlow model
MODEL_DIR = '/home/andrey/msds_practicum/msds_practicum/src/models/trained_models/Model_0002_CNN_ResNet__loss_sparse_categorical_crossentropy'
model = tf.keras.models.load_model(MODEL_DIR)

def capture_frame(buffer, width, height):
    success, map_info = buffer.map(Gst.MapFlags.READ)
    if not success:
        raise RuntimeError('Could not map buffer for reading')
    frame = np.frombuffer(map_info.data, dtype=np.uint8, count=width * height * 3 // 2)
    buffer.unmap(map_info)
    frame = cv2.cvtColor(frame.reshape(height * 3 // 2, width), cv2.COLOR_YUV2RGB_NV12)
    frame = cv2.resize(frame, (224, 224))
    return frame

def preprocess_image(image):
    image = image.astype('float32') / 255.0
    return image

def perform_prediction(image_np):
    prediction = model.predict(np.expand_dims(image_np, axis=0))
    print(f"Prediction: {prediction}")

def on_new_sample(appsink):
    sample = appsink.emit('pull-sample')
    if isinstance(sample, Gst.Sample):
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = structure.get_value('width')
        height = structure.get_value('height')
        frame = capture_frame(buffer, width, height)
        preprocessed_frame = preprocess_image(frame)
        perform_prediction(preprocessed_frame)
    return Gst.FlowReturn.OK

# Create GStreamer elements
source = Gst.ElementFactory.make('nvarguscamerasrc', 'source')
caps_filter = Gst.ElementFactory.make('capsfilter', 'caps_filter')
caps_filter.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM), width=1280, height=720, framerate=21/1, format=NV12"))
vidconv1 = Gst.ElementFactory.make('nvvidconv', 'vidconv1')
vidconv2 = Gst.ElementFactory.make('nvvidconv', 'vidconv2')
transform = Gst.ElementFactory.make('nvegltransform', 'transform')
sink = Gst.ElementFactory.make('nveglglessink', 'sink')
appsink = Gst.ElementFactory.make('appsink', 'appsink')  # Moved here

# Configure appsink
appsink.set_property('emit-signals', True)
appsink.connect('new-sample', on_new_sample)

# Pipeline setup
pipeline = Gst.Pipeline.new('test-pipeline')
for element in [source, caps_filter, vidconv1, vidconv2, transform, sink, appsink]:  # Add appsink to pipeline
    if not element:
        print("Failed to create elements")
        exit(-1)
    pipeline.add(element)

# Element properties
source.set_property('sensor_id', 0)
vidconv1.set_property('flip-method', 2)
caps_filter.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM), width=3280, height=2464, framerate=21/1, format=NV12"))

# Link elements
elements = [source, caps_filter, vidconv1, vidconv2, transform, sink]  # Removed appsink from linking
for i in range(len(elements)-1):
    if not elements[i].link(elements[i+1]):
        print(f"ERROR: Could not link {elements[i].get_name()} to {elements[i+1].get_name()}")
        exit(-1)

# Set the pipeline to "PLAYING" state
print("Starting pipeline")
ret = pipeline.set_state(Gst.State.PLAYING)
if ret == Gst.StateChangeReturn.FAILURE:
    print("Unable to set the pipeline to the playing state.")
    exit(-1)

# Main loop
loop = GLib.MainLoop()
try:
    loop.run()
except KeyboardInterrupt:
    print('interrupted by user')
finally:
    pipeline.set_state(Gst.State.NULL)