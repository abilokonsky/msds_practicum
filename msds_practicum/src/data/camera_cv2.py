import cv2

# Open the camera device
cap = cv2.VideoCapture('/dev/video1')  # Adjust the device number as necessary

# Capture a frame
ret, frame = cap.read()

if ret:
    # Save the frame as an image file
    cv2.imwrite('captured_image.png', frame)
else:
    print("Failed to capture image")

cap.release()