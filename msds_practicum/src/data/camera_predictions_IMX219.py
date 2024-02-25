import gi
import cv2
import threading
import numpy as np
import tensorflow as tf
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Initialize GStreamer
Gst.init(None)

pipeline_description = ("nvarguscamerasrc ! "
                        "video/x-raw(memory:NVMM),width=3820, height=2464, framerate=21/1, format=NV12 ! "
                        "nvvidconv flip-method=0 ! video/x-raw,width=960, height=616 ! "
                        "videoconvert ! video/x-raw, format=(string)BGR ! appsink name=mysink emit-signals=true max-buffers=1 drop=true")


pipeline = Gst.parse_launch(pipeline_description)

# Load your TensorFlow model
model = tf.keras.models.load_model('/home/andrey/msds_practicum/msds_practicum/src/models/trained_models/Model_0002_CNN_ResNet__loss_sparse_categorical_crossentropy')

appsink = pipeline.get_by_name("mysink")
if not appsink:
    raise RuntimeError("Failed to find appsink in pipeline")

# Function to be called on new sample
def on_new_sample(sink, data):
    sample = sink.emit("pull-sample")
    if sample:
        # Retrieve the buffer
        buffer = sample.get_buffer()
        # Extract data
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        # Assuming BGR format, create an ndarray
        frame = np.ndarray(
            shape=(616, 960, 3),
            buffer=map_info.data,
            dtype=np.uint8
        )
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize the frame to 256x256
        frame_resized = cv2.resize(frame_rgb, (256, 256))

        # Perform prediction
        prediction = model.predict(np.expand_dims(frame_resized, axis=0))
        print("Prediction:", prediction)
        
        buffer.unmap(map_info)

        return Gst.FlowReturn.OK
    return Gst.FlowReturn.ERROR

# Connect the new-sample signal of the appsink
appsink.connect("new-sample", on_new_sample, None)

# Start the pipeline
pipeline.set_state(Gst.State.PLAYING)

# Keep the script running so the pipeline doesn't immediately exit
try:
    loop = GLib.MainLoop()
    loop.run()
except KeyboardInterrupt:
    loop.quit()
    pipeline.set_state(Gst.State.NULL)