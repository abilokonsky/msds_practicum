import cv2
import numpy as np

def capture_video(device_id, width, height, fps):
    # Open the video capture device
    cap = cv2.VideoCapture(device_id)

    # Set capture properties, if supported
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'RG10'))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert Bayer RG10 to BGR for display
        # Note: Adjust the conversion method based on your specific Bayer format if needed
        bgr = cv2.cvtColor(frame, cv2.COLOR_BayerRG2BGR)

        cv2.imshow('Video', bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Update the device ID (0 for default) and desired resolution and FPS
    capture_video(device_id=1, width=1920, height=1080, fps=30)