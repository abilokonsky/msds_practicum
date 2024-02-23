import gi
gi.require_version('Gst', '1.0')    
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

    # Diagnostic logging
    print(f"Actual buffer size: {map_info.size}")
    expected_size = width * height * 3 // 2
    print(f"Expected buffer size (NV12): {expected_size}")

    if map_info.size < expected_size:
        raise RuntimeError(f'Buffer is smaller than expected size: {map_info.size} < {expected_size}')
    
    # Assuming the buffer size is correct, proceed with frame extraction
    try:
        frame = np.frombuffer(map_info.data, dtype=np.uint8, count=map_info.size)
        frame = cv2.cvtColor(frame.reshape(height + height // 2, width), cv2.COLOR_YUV2BGR_NV12)
    except Exception as e:
        print(f"Error processing frame: {e}")
        frame = None
    finally:
        buffer.unmap(map_info)
    
    return frame


def preprocess_image(image):
    image = image.astype('float32') / 255.0
    return image

def perform_prediction(image_np):
    prediction = model.predict(image_np.reshape(1, 256, 256, 3))
    return prediction

def overlay_prediction_on_frame(frame, prediction, position, window_size=(256, 256)):
    # Convert prediction to text
    prediction_text = str(prediction)
    
    # Define position for text (e.g., top-left corner of the window)
    text_position = (position[0], position[1] + 20)  # Adjust based on your needs
    
    # Overlay text on the frame
    cv2.putText(frame, prediction_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (255, 0, 0), 2)
    

def process_rolling_windows(frame, window_size=(256, 256)):
    print("processing rolling windows")
    height, width, _ = frame.shape
    stride = 128  # Or any stride you're using
    
    for y in range(0, height - window_size[1] + 1, stride):
        for x in range(0, width - window_size[0] + 1, stride):
            window = frame[y:y + window_size[1], x:x + window_size[0]]
            preprocessed_window = preprocess_image(window)
            
            # Perform prediction and get the result
            prediction = perform_prediction(preprocessed_window)
            
            # Overlay the prediction on the original frame
            overlay_prediction_on_frame(frame, prediction, (x, y), window_size)

def on_new_sample(appsink):
    print("Sample received")  # Debug print
    sample = appsink.emit('pull-sample')
    if isinstance(sample, Gst.Sample):
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = structure.get_value('width')
        height = structure.get_value('height')

        frame = capture_frame(buffer, width, height)
        process_rolling_windows(frame)

    return Gst.FlowReturn.OK



# Create GStreamer pipeline
pipeline = Gst.Pipeline.new("my-pipeline")

# Create and configure elements
source = Gst.ElementFactory.make("nvarguscamerasrc", "source")
capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
# Example: Set to 1080p at 30 fps for better performance in real-time applications
capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1, format=NV12"))
tee = Gst.ElementFactory.make("tee", "tee")
queue1 = Gst.ElementFactory.make("queue", "queue1")
queue2 = Gst.ElementFactory.make("queue", "queue2")
vidconv1 = Gst.ElementFactory.make("nvvidconv", "vidconv1")
vidconv2 = Gst.ElementFactory.make("nvvidconv", "vidconv2")
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