import gi
import cv2
import numpy as np
import tensorflow as tf
from gi.repository import Gst, GLib

# Initialize GStreamer
Gst.init(None)

# Load the TensorFlow model
MODEL_DIR = '../models/trained_models/Model_0002_CNN_ResNet__loss_sparse_categorical_crossentropy'
model = tf.keras.models.load_model(MODEL_DIR)

def capture_frame(buffer, width, height):
    # Memory mapping the buffer to access the data
    success, map_info = buffer.map(Gst.MapFlags.READ)
    if not success:
        raise RuntimeError('Could not map buffer for reading')

    # Create a NumPy array from the buffer data
    # NV12 is 12 bits per pixel, so the size is width * height * 1.5
    # However, we are reading 8 bits at a time (np.uint8), so the size in the frombuffer should be width * height * 3 // 2
    frame = np.frombuffer(map_info.data, dtype=np.uint8, count=width * height * 3 // 2)

    # Unmap the buffer when done
    buffer.unmap(map_info)

    # Convert NV12 data to RGB (cv2.cvtColor expects the frame height to be 1.5 times the NV12 height because of the UV plane)
    frame = cv2.cvtColor(frame.reshape(height * 3 // 2, width), cv2.COLOR_YUV2RGB_NV12)

    # The frame can be further processed (e.g., resized) if necessary
    frame = cv2.resize(frame, (256, 256))  # Resize to the input shape expected by the model

    return frame

def preprocess_image(image):
    # Preprocess the image as required by the model
    image = image.astype('float32') / 255.0
    return image

def perform_prediction(image_np):
    # Use the preprocessed image to perform a prediction
    prediction = model.predict(np.expand_dims(image_np, axis=0))
    # Handle the prediction
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


# Create the elements
source = Gst.ElementFactory.make('nvarguscamerasrc', 'source')
caps_filter = Gst.ElementFactory.make('capsfilter', 'caps_filter')
caps_filter.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM), width=3280, height=2464, framerate=21/1, format=NV12"))
vidconv1 = Gst.ElementFactory.make('nvvidconv', 'vidconv1')
vidconv2 = Gst.ElementFactory.make('nvvidconv', 'vidconv2')
transform = Gst.ElementFactory.make('nvegltransform', 'transform')
sink = Gst.ElementFactory.make('nveglglessink', 'sink')

# Set properties
source.set_property('sensor_id', 0)
vidconv1.set_property('flip-method', 2)

# Create the empty pipeline
pipeline = Gst.Pipeline.new('test-pipeline')

if not pipeline or not source or not caps_filter or not vidconv1 or not vidconv2 or not transform or not sink:
    print("Not all elements could be created.")
    exit(-1)

# Build the pipeline
pipeline.add(source)
pipeline.add(caps_filter)
pipeline.add(vidconv1)
pipeline.add(vidconv2)
pipeline.add(transform)
pipeline.add(sink)


# Link the elements one after the other
if not source.link(caps_filter):
    print("ERROR: Could not link source to caps_filter")
    exit(-1)
if not caps_filter.link(vidconv1):
    print("ERROR: Could not link caps_filter to vidconv1")
    exit(-1)
if not vidconv1.link(vidconv2):
    print("ERROR: Could not link vidconv1 to vidconv2")
    exit(-1)
if not vidconv2.link(transform):
    print("ERROR: Could not link vidconv2 to transform")
    exit(-1)
if not transform.link(sink):
    print("ERROR: Could not link transform to sink")
    exit(-1)

# Set the pipeline to "PLAYING" state
print("Starting pipeline")
pipeline = Gst.Pipeline.new('test-pipeline')

appsink = Gst.ElementFactory.make('appsink', 'appsink')
appsink.set_property('emit-signals', True)
appsink.connect('new-sample', on_new_sample)

# Add and link the appsink element just before the sink element
pipeline.add(appsink)
vidconv2.link(appsink)
appsink.link(sink)

# Start the GStreamer main loop
loop = GLib.MainLoop()
try:
    loop.run()
except KeyboardInterrupt:
    pass
finally:
    pipeline.set_state(Gst.State.NULL)