import cv2
import numpy as np
import gi
import time

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Initialize GStreamer
Gst.init(None)

class FrameSaver:
    def __init__(self):
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

            # Save the frame as PNG
            cv2.imwrite(f"frame-{time.time()}.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            buffer.unmap(map_info)

        return Gst.FlowReturn.OK

    def run(self, duration):
        # Run the loop for a specified duration (seconds)
        end_time = time.time() + duration
        while time.time() < end_time:
            time.sleep(0.5)  # Adjust the frequency of captures

# Main execution
if __name__ == "__main__":
    saver = FrameSaver()
    saver.run(10)  # Run for 10 seconds
