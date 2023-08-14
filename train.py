import os
import time

import matplotlib.pyplot as plt
import numpy as np
import resources.utilities as ut
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential

osp = os.path

# This will resize the pngs for training
# ------------------------------------------------------------------------
# Common Practice: Many popular pre-trained models and architectures
# have been designed and fine-tuned for standard image sizes like 224x224
# or 299x299. Using similar dimensions might make it easier to leverage
# pre-trained models.
# ------------------------------------------------------------------------
image_height, image_width = 299, 299

# Read the image data and the csv results
DATASET_IDS = ['DS_00001_03777', 'DS_10001_14130', 'DS_14131_17630', 'DS_20001_23642']

tref = time.time()
images, values = ut.load_datasets(DATASET_IDS, img_size=(image_height, image_width))
ut.print_ok(f'Succesfully loaded {len(images)} data points ({time.time()-tref:.1f}s)')


# Define the model architecture and compile it
ut.print_('info', 'Defining and compiling model', 'bold')
tref = time.time()
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 1)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(3))  # Output layer with 3 neurons for mx, my, mxy

model.compile(optimizer='adam', loss='mean_squared_error')
ut.print_ok(f'Model compiled ({time.time()-tref:.1f}s)')


# Organize data into training and testing sets
ut.print_('info', f'Organizing data as 20% tests and 80% training', 'bold')
tref = time.time()
splitted = train_test_split(images, values, test_size=0.2, random_state=42)
train_images, test_images, train_values, test_values = splitted

# Train the model
epochs = 20
batch_size = 64


# Define a custom callback to plot real-time training loss
class PlotLossCallback(Callback):
    def on_train_begin(self, logs=None):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs['loss'])
        self.val_losses.append(logs.get('val_loss'))
        self.plot_losses()

    def plot_losses(self):
        plt.figure(figsize=(10, 6))  # Set the figure size

        plt.plot(self.losses, label='Training Loss', color='blue', linestyle='dashed', linewidth=2)
        plt.plot(self.val_losses, label='Validation Loss', color='red', linewidth=2)

        plt.title('Training and Validation Loss', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend(fontsize=12)

        plt.grid(True, linestyle='dotted', alpha=0.7)  # Add grid lines

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.tight_layout()
        plt.savefig("losses.png")


# Create an instance of the custom callback
plot_loss_callback = PlotLossCallback()

ut.print_('info', f'Training the model', 'bold')
tref = time.time()
model.fit(train_images, train_values,
          epochs=epochs, batch_size=batch_size,
          validation_data=(test_images, test_values),
          callbacks=[plot_loss_callback])
ut.print_ok(f'Done ({time.time()-tref:.1f}s)')

# Evaluate the model (optional as it is already given in model.fit)
test_loss = model.evaluate(test_images, test_values)

# # Predict values for new images
# new_images = preprocess_new_images(new_png_images)
# predicted_values = model.predict(new_images)

# Save model architecture and weights
model.save('model_15kpts_2023-08-14.h5')

# # Optionally, export to ONNX format
# onnx_model = onnxmltools.convert_keras(model)
# onnx.save_model(onnx_model, 'model_name.onnx')

# # Test the exported model
# loaded_model = tf.keras.models.load_model('added_masses_hb_2023-08-14.h5')
# test_prediction = loaded_model.predict(test_data)



# For predicting
# ------------------
# images, values = ut.load_dataset('DS_00001_03777')
# pred = model.predict(images)
# out = np.column_stack(pred, values)
# df = pd.DataFrame(out, columns=['MX-pred', 'MY-pred', 'MXY-pred', 'MX-calc', 'MY-calc', 'MXY-calc'])
# df = df[['MX-pred', 'MX-calc', 'MY-pred', 'MY-calc', 'MXY-pred', 'MXY-calc']]
