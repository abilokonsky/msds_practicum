import cv2
import numpy as np
import tensorflow as tf

# Load the saved TensorFlow model
model = tf.keras.models.load_model('path_to_your_saved_model')

def preprocess_frame(frame):
    # Resize frame to match the input shape expected by the model (256x256)
    resized_frame = cv2.resize(frame, (256, 256))
    # Convert frame to RGB format
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    # Normalize pixel values to range [0, 1]
    normalized_frame = rgb_frame / 255.0
    return normalized_frame

def show_camera():
    window_title = "CSI Camera"

    print(gstreamer_pipeline(flip_method=0))
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            while True:
                ret_val, frame = video_capture.read()
                if ret_val:
                    # Preprocess the camera frame
                    preprocessed_frame = preprocess_frame(frame)
                    # Perform inference using the model
                    predictions = model.predict(np.expand_dims(preprocessed_frame, axis=0))
                    # Display the camera frame and predictions
                    cv2.imshow(window_title, frame)
                    print(predictions)  # Modify as needed to display predictions
                else:
                    break 
                keyCode = cv2.waitKey(10) & 0xFF
                if keyCode == 27 or keyCode == ord('q'):
                    break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")

if __name__ == "__main__":
    show_camera()