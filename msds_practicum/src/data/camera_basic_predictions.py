import cv2
import tensorflow as tf
import numpy as np

# Load your trained model
model = tf.keras.models.load_model('path_to_your_model.h5')

def process_frame(frame):
    # Preprocess the frame to match your model's input requirements.
    # This might include resizing, normalization, expanding dimensions, etc.
    # Adjust the preprocessing steps accordingly.
    frame = cv2.resize(frame, (128, 128))  # Resize to model expected input
    frame = frame / 255.0  # Normalize
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

def capture_from_camera():
    # Define the GStreamer pipeline for the MIPI CSI camera
    # Adjust the capture width, height, and framerate as needed
    cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, width=(int)128, height=(int)128, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Preprocess the frame
            processed_frame = process_frame(frame)

            # Perform inference
            prediction = model.predict(processed_frame)
            
            # Here, add code to display the prediction or further process it
            print("Prediction: ", prediction)

            # Display the resulting frame (optional)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    capture_from_camera()