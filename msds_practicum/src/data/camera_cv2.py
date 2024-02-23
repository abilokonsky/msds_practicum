import cv2
import numpy as np
import gi
import time
import tensorflow as tf

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Initialize GStreamer
Gst.init(None)

class PredictionProcessor:
    def __init__(self, class_names, model_path):
        self.class_names = class_names
        self.model = tf.keras.models.load_model(model_path)

    def process(self, frame):
        # Assuming `frame` is an RGB image of shape (height, width, 3)
        # Preprocess the image
        preprocessed_frame = cv2.resize(frame, (256, 256))  # Resize the frame to match model input
        preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0) / 255.0  # Normalize

        # Perform prediction
        prediction = self.model.predict(preprocessed_frame)
        top_indices = prediction.flatten().argsort()[-3:][::-1]

        # Retrieve the class names and probabilities for the top predictions
        top_classes = [self.class_names[i] for i in top_indices]
        top_probabilities = prediction.flatten()[top_indices]

        # Return a list of tuples containing class names and their probabilities
        return list(zip(top_classes, top_probabilities))

class FrameSaver:
    def __init__(self, processor):
        self.processor = processor
        # Define the GStreamer pipeline
        self.pipeline = Gst.parse_launch(
            "v4l2src ! videoconvert ! videoscale ! "
            "video/x-raw,format=RGB,width=640,height=480 ! appsink name=sink emit-signals=True"
        )
        self.appsink = self.pipeline.get_by_name("sink")
        self.appsink.connect("new-sample", self.on_new_sample, self.appsink)
        self.pipeline.set_state(Gst.State.PLAYING)

    def on_new_sample(self, sink, data):
        sample = sink.emit("pull-sample")
        if sample:
            buffer = sample.get_buffer()
            caps_format = sample.get_caps().get_structure(0)
            width = caps_format.get_value('width')
            height = caps_format.get_value('height')

            # Extract frame data
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if not success:
                return Gst.FlowReturn.ERROR

            frame = np.ndarray(
                shape=(height, width, 3),
                dtype=np.uint8,
                buffer=map_info.data
            )

            buffer.unmap(map_info)

            # Display the frame
            cv2.imshow('Captured Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return Gst.FlowReturn.EOS  # End of Stream

            # Process the frame for prediction
            predictions = self.processor.process(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            for class_name, probability in predictions:
                print(f"{class_name} certainty = {probability*100:.2f}% ")

        return Gst.FlowReturn.OK

    def run(self, duration):
        # Run the loop for a specified duration (seconds)
        end_time = time.time() + duration
        while time.time() < end_time:
            time.sleep(0.5)  # Adjust the frequency of captures
        cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    class_names = [f"{5 * i} degrees" for i in range(1, 20)]
    model_path = '/home/andrey/msds_practicum/msds_practicum/src/models/trained_models/Model_0002_CNN_ResNet__loss_sparse_categorical_crossentropy'
    processor = PredictionProcessor(class_names, model_path)
    saver = FrameSaver(processor)
    saver.run(10)  # Run for 10 seconds
