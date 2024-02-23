import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

# Initialize GStreamer
Gst.init(None)

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
pipeline.set_state(Gst.State.PLAYING)


# Start playing
ret = pipeline.set_state(Gst.State.PLAYING)
if ret == Gst.StateChangeReturn.FAILURE:
    print("Unable to set the pipeline to the playing state.")
    exit(-1)

# Wait until error or EOS
bus = pipeline.get_bus()
msg = bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS)


# Parse message
if msg:
    if msg.type == Gst.MessageType.ERROR:
        err, debug = msg.parse_error()
        print(f"Error received from element {msg.src.get_name()}: {err}")
        print(f"Debugging information: {debug}")
    elif msg.type == Gst.MessageType.EOS:
        print("End-Of-Stream reached.")

# Free resources
pipeline.set_state(Gst.State.NULL)