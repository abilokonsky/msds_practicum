import cv2
import numpy as np
import tensorflow as tf

# Load the saved TensorFlow model
model = tf.keras.models.load_model('models/trained_models/Model_0001_CNN_ResNet__loss_sparse_categorical_crossentropy')

# Function to preprocess the image for your model
def preprocess_image(image):
    # Resize to (256, 256) and normalize
    image = tf.image.resize(image, (256, 256))
    image = image / 255.0
    return image

def process_frame(frame):
    # Define the region of interest (ROI) coordinates
    x, y, w, h = 100, 100, 256, 256  # Example coordinates (adjust as needed)

    # Crop the ROI from the frame
    roi = frame[y:y+h, x:x+w]

    # Preprocess the ROI for your model
    input_image = preprocess_image(roi)

    # Perform inference using your model
    predictions = model.predict(np.expand_dims(input_image, axis=0))

    # Process predictions (e.g., visualize or use for decision making)

    return predictions

def show_camera():
    window_title = "CSI Camera"

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            while True:
                ret_val, frame = video_capture.read()
                if not ret_val:
                    print("Error: Unable to read frame from camera")
                    break
                
                # Process the frame to perform computer vision tasks
                predictions = process_frame(frame)

                # Visualize the predictions on the frame (optional)
                # Example: draw bounding boxes, display class labels, etc.

                # Display the processed frame
                cv2.imshow(window_title, frame)
                
                # Check for user input to exit
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