import gi
import cv2
import numpy as np
import tensorflow as tf
from threading import Thread
import queue
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Initialize GStreamer
Gst.init(None)

# Define the GStreamer pipeline
pipeline_description = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM),width=3820, height=2464, framerate=21/1, format=NV12 ! "
    "nvvidconv flip-method=0 ! video/x-raw,width=960, height=616 ! "
    "tee name=t "
    "t. ! queue ! videoconvert ! video/x-raw, format=(string)BGR ! appsink name=mysink emit-signals=true max-buffers=1 drop=true "
    "t. ! queue ! videoconvert ! xvimagesink sync=false"
)
pipeline = Gst.parse_launch(pipeline_description)

# Load your TensorFlow model
model = tf.keras.models.load_model('/home/andrey/msds_practicum/msds_practicum/src/models/trained_models/Model_0002_CNN_ResNet__loss_sparse_categorical_crossentropy')

appsink = pipeline.get_by_name("mysink")

# Frame processing queue
frame_queue = queue.Queue(maxsize=10)


class PredictionProcessor:
    def __init__(self, class_names):
        self.class_names = class_names

    def process(self, prediction):
        # Assuming `prediction` is a numpy array of shape (1, num_classes)
        # Flatten the prediction to simplify indexing
        prediction = prediction.flatten()
        
        # Get indices of the top 3 predictions in descending order
        top_indices = np.argsort(prediction)[::-1][:1]
        
        # Retrieve the class names and probabilities for the top 3 predictions
        top_classes = [self.class_names[i] for i in top_indices]
        top_probabilities = prediction[top_indices]
        
        # Return a list of tuples containing class names and their probabilities
        return list(zip(top_classes, top_probabilities))

class_names = [f"{5 * i} degrees" for i in range(1, 20)]  # Adjust to match the number of your classes
processor = PredictionProcessor(class_names)




def perform_prediction(image_np):

    prediction_array = np.array([model.predict(np.expand_dims(image_np, axis=0))], dtype=float)
    top_predictions = processor.process(prediction_array)
    for i, (class_name, probability) in enumerate(top_predictions, start=1):
        prediction_str = str(f"{class_name} certainty = {probability*100:.2f}% " )

    return prediction_str

def frame_processor():
    while True:
        frame_rgb = frame_queue.get()
        if frame_rgb is None:  # Termination signal
            break

        # Your existing sliding window and prediction logic
        window_size = (256, 256)  # Window size
        stride = 64  # Stride for sliding the window
        for y in range(0, frame_rgb.shape[0] - window_size[1] + 1, stride):
            for x in range(0, frame_rgb.shape[1] - window_size[0] + 1, stride):
                window = frame_rgb[y:y + window_size[1], x:x + window_size[0]]
                
                prediction = model.predict(np.expand_dims(window, axis=0))
                top_predictions = processor.process(prediction)
                prediction_str = " | ".join([f"{class_name}: {probability*100:.2f}%" for class_name, probability in top_predictions])

                # Example: Print or process the prediction for each window
                print(f"Window [{x}, {y}] Prediction: {prediction_str}")

# Start the frame processing thread
processing_thread = Thread(target=frame_processor)
processing_thread.start()

def on_new_sample(sink, data):
    sample = sink.emit("pull-sample")
    if sample:
        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        # Assuming BGR format, create an ndarray
        frame = np.ndarray(shape=(616, 960, 3), buffer=map_info.data, dtype=np.uint8)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if not frame_queue.full():
            frame_queue.put(frame_rgb)  # Add frame to processing queue

        buffer.unmap(map_info)
        return Gst.FlowReturn.OK
    return Gst.FlowReturn.ERROR

# Connect the new-sample signal of the appsink
appsink.connect("new-sample", on_new_sample, None)

# Start the pipeline
pipeline.set_state(Gst.State.PLAYING)

# Main loop
try:
    loop = GLib.MainLoop()
    loop.run()
except KeyboardInterrupt:
    # Cleanup
    frame_queue.put(None)  # Send termination signal to the processing thread
    processing_thread.join()
    loop.quit()
    pipeline.set_state(Gst.State.NULL)
