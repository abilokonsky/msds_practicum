import subprocess
import time
import os

image_path = 'images/captured_image.jpg'  # Ensure this path is accessible and writable

# Start 'feh' to display the image; it will auto-refresh when the file changes.
# Use '--reload' with a large number to effectively pause on the current image.
feh_process = subprocess.Popen(['feh', '--reload', '99999', image_path])

try:
    while True:
        # Capture an image
        # Ensure 'nvgstcapture-1.0' saves the image to 'image_path'
        subprocess.run(['nvgstcapture-1.0', '--image-res=8', f'--file-name={image_path}'])

        print("Image captured.")

        # 'feh' automatically reloads the image when the file changes.
        # Wait for a few seconds before capturing the next image
        time.sleep(2)  # Adjust the sleep time as needed

except KeyboardInterrupt:
    print("Stopped by user.")
finally:
    # Clean up: terminate 'feh' when done
    feh_process.terminate()