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
    frame = cv2.resize(frame, (256, 256))
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

# Create GStreamer pipeline
pipeline = Gst.Pipeline.new("my-pipeline")

# Create and configure elements
source = Gst.ElementFactory.make("nvarguscamerasrc", "source")
tee = Gst.ElementFactory.make("tee", "tee")
queue1 = Gst.ElementFactory.make("queue", "queue1")
queue2 = Gst.ElementFactory.make("queue", "queue2")
vidconv1 = Gst.ElementFactory.make("nvvidconv", "vidconv1")
vidconv2 = Gst.ElementFactory.make("nvvidconv", "vidconv2")
capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
appsink = Gst.ElementFactory.make("appsink", "appsink")
transform = Gst.ElementFactory.make("nvegltransform", "transform")
sink = Gst.ElementFactory.make("nveglglessink", "sink")

if not all([source, tee, queue1, queue2, vidconv1, vidconv2, capsfilter, appsink, transform, sink]):
    print("Failed to create pipeline elements")
    exit(-1)

# Set properties
source.set_property("sensor_id", 0)
capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1, format=NV12"))
appsink.set_property("emit-signals", True)
appsink.connect("new-sample", on_new_sample)

# Add elements to the pipeline
for element in [source, tee, queue1, vidconv1, appsink, queue2, vidconv2, transform, sink]:
    pipeline.add(element)

# Link elements
source.link(tee)
tee.link(queue1)
queue1.link(vidconv1)
vidconv1.link(appsink)
tee.link(queue2)
queue2.link(vidconv2)
vidconv2.link(transform)
transform.link(sink)

# Start the pipeline
pipeline.set_state(Gst.State.PLAYING)

# Main loop
loop = GLib.MainLoop()
try:
    loop.run()
except KeyboardInterrupt:
    print('Interrupted by user')
finally:
    pipeline.set_state(Gst.State.NULL)