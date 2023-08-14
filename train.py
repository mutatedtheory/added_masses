import os

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential

osp = os.path

image_height, image_width = 1600, 1600

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, num_channels)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(3))  # Output layer with 3 neurons for speed, steering angle, and brake pressure

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Read the image data and the csv results
DATASET_ID = 'DS_10001_14130'
images_dir = osp.join('output', DATASET_ID, 'images')
calcs_dir = osp.join('output', DATASET_ID, 'calcs')

fns = list(os.listdir(calcs_dir))
fns.sort()

# Read simulation results
images, values = [], []
for fn in fns:
    if fn.endswith('.csv'):
        csv_fn = os.path.join(calcs_dir, fn)
        png_fn = osp.join(images_dir, os.path.splitext(fn)[0] + ".png")

        data = np.loadtxt(csv_fn, delimiter=' ')
        values.append(np.array((data[0, 0], data[1, 1], data[0, 1])))

        img = Image.open(png_fn)
        # img = img.resize((image_width, image_height))  # Resize the image to desired dimensions
        img_array = np.array(img)[:, :, 0] / 255.0  # Normalize pixel values to [0, 1]
        images.append(img_array)


# Step 3: Organize data into training and testing sets
train_images, test_images, train_values, test_values = train_test_split(
    images, values, test_size=0.2, random_state=42)


epochs = 20
batch_size = 64

# Train the model
model.fit(train_images, train_values, epochs=epochs, batch_size=batch_size, validation_data=(test_images, test_values))

# Evaluate the model
test_loss = model.evaluate(test_images, test_values)

# Predict values for new images
new_images = preprocess_new_images(new_png_images)
predicted_values = model.predict(new_images)
