import subprocess
import time

try:
    while True:
        # Capture an image
        subprocess.run(['nvgstcapture-1.0', '--image-res=8', '--file-name=captured_image.jpg'])

        print("Image captured.")

        # Wait for a few seconds before capturing the next image
        time.sleep(2)  # Adjust the sleep time as needed

except KeyboardInterrupt:
    print("Stopped by user.")