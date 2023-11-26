# Import(s)
import os
import cv2
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Tensorflow disable debug logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Establish letter dict, path, and read data from CSV
word_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}
dir_path = os.path.dirname(os.path.realpath(__file__))
parent = os.path.abspath(os.path.join(dir_path, os.pardir))
data = pd.read_csv(parent+"/CharCNN/data/a-z_handwritten_data.csv").astype('float32')

# Split and reshape data for the model
X = data.drop('0', axis=1)
y = data['0']

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
train_x = np.reshape(train_x.values, (train_x.shape[0], 28, 28))
test_x = np.reshape(test_x.values, (test_x.shape[0], 28, 28))

# Put labels as categorial data
test_yOHE = to_categorical(test_y, num_classes=26, dtype='int')
train_yOHE = to_categorical(train_y, num_classes=26, dtype='int')

# Reshape train/test input for the model
# noinspection PyArgumentList
test_X = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1)
# noinspection PyArgumentList
train_X = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)

### CNN model implementation ###

# Sequential (1 input tensor and 1 output tensor)
# (The tensors in this case are matrices)
model = Sequential()

# Add convolution layers (2D matrix kernel) and maxpool (avoids over-scanning)
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

# Flatten to connect convolution layers to dense layers
model.add(Flatten())

# Add Dense layers (last as softmax for prb)
model.add(Dense(64, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(26, activation="softmax"))

### ------------------------ ###

# Compile and save model to H5
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

history = model.fit(train_X, train_yOHE, epochs=1, callbacks=[reduce_lr, early_stop],
validation_data=(test_X, test_yOHE))

model.save('cnn_model.h5')

# Accuracies and losses
print("Validation accuracy:", history.history['val_accuracy'])
print("Validation loss:", history.history['val_loss'])
print("Training accuracy:", history.history['accuracy'])
print("Training loss:", history.history['loss'])

### Prediction for a sample image ###

# Import and make copy
img = cv2.imread(parent+"/CharCNN/test_images/image_C.jpg")
imgc = img.copy()

# Resize for window and change color space
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (400, 440))

# Grayscale image
img_gray = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY)
_, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

# Final resize and reshape
img_final = cv2.resize(img_thresh, (28, 28))
img_final = np.reshape(img_final, (1, 28, 28, 1))

# Get prediction of the image
img_pred = word_dict[np.argmax(model.predict(img_final))]

# Add text and display
cv2.putText(img, "  English character recognition", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color=(0, 0, 0))
cv2.putText(img, "  Prediction: " + img_pred, (20, 410), cv2.FONT_HERSHEY_DUPLEX, 1.3, color=(255, 0, 0))
cv2.imshow('CharCNN', img)
print("(FOCUS ON CV2 WINDOW)")

### ----------------------------- ###

# Keep window until ESC is pressed
while True:
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
