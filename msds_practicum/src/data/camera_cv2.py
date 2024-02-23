import cv2
import time

# Initialize the camera
# The device number might vary (often 0 or 1 for built-in/webcams)
cap = cv2.VideoCapture(0)  

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            # Save the captured image to disk
            cv2.imwrite('captured_image.png', frame)
            print("Image captured and saved.")

        # Wait for a few seconds before capturing the next image
        time.sleep(2)  # Adjust the sleep time as needed

finally:
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
