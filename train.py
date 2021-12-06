import tensorflow as tf
import os
import argparse
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

help_msg = "This loads in a trained modeal and returns a prediction"
parser = argparse.ArgumentParser(description=help_msg)
parser.add_argument("-e",
                    "--epochs",
                    type=int,
                    help="Number of epochs")
parser.add_argument("-d",
                    "--data_path",
                    type=str,
                    help="Path to the data source")
args = parser.parse_args()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,
                           (3, 3),
                           activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1/255,
                                   validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
        args.data_path,
        target_size=(300, 300),
        batch_size=32,
        class_mode='binary',
        subset='training')

validation_generator = train_datagen.flow_from_directory(
        args.data_path,
        target_size=(300, 300),
        batch_size=32,
        class_mode='binary',
        subset='validation')

history = model.fit(
      train_generator,
      steps_per_epoch=train_generator.samples // 32,
      epochs=args.epochs,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples // 32,
      verbose=1)

print('cnvrg_tag_test_accuracy: ', history.history['val_acc'][-1])
print('cnvrg_tag_test_loss: ', history.history['val_loss'][-1])
if not os.path.exists('output'):
    os.mkdir('output')
model.save('output/imagizer.model.h5')
