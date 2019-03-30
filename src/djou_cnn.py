import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout

model = keras.models.Sequential()

model.add(Conv2D(64, 5, input_shape=(160, 255, 1), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))

model.compile(loss='categorical_crossentropy',
             optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
