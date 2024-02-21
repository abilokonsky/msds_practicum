import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import glob

# Assuming your images are stored in a directory structure:
# - images/
#   - rgb/
#   - iseg/

data_dir = "path/to/your/images/directory"
rgb_dir = os.path.join(data_dir, 'rgb')
iseg_dir = os.path.join(data_dir, 'iseg')

# List of image files
rgb_images = glob.glob(os.path.join(rgb_dir, '*.png'))
iseg_images = glob.glob(os.path.join(iseg_dir, '*.png'))

# Example function to load and preprocess images
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)  # Adjust channels accordingly
    image = tf.image.resize(image, [256, 256])  # Resize images if needed
    image /= 255.0  # Normalize to [0,1]
    return image

# Example function to create a dataset from image paths
def create_dataset(image_paths, label_paths):
    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    image_dataset = image_dataset.map(load_and_preprocess_image)
    
    label_dataset = tf.data.Dataset.from_tensor_slices(label_paths)
    label_dataset = label_dataset.map(load_and_preprocess_image)  # Assuming labels are images too
    
    return tf.data.Dataset.zip((image_dataset, label_dataset))

# Create dataset
dataset = create_dataset(rgb_images, iseg_images)

# Split dataset into training and validation
train_size = int(0.8 * len(rgb_images))
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

# Batch and prefetch for performance
BATCH_SIZE = 32
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)






model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    # Add more layers as needed
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Adjust the output layer according to your task
])




def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_flip_up_down(image)
    return image, label



# Apply the augmentation to the training dataset
train_dataset = train_dataset.map(augment)



model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Choose loss function based on your problem
              metrics=['accuracy'])

history = model.fit(train_dataset, validation_data=val_dataset, epochs=10)




val_loss, val_acc = model.evaluate(val_dataset)
print("Validation Loss: ", val_loss)
print("Validation Accuracy: ", val_acc)


