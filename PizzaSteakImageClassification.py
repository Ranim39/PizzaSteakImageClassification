
import zipfile
!wget https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip
zip_ref = zipfile.ZipFile("pizza_steak.zip")
zip_ref.extractall()
zip_ref.close()

!ls pizza_steak/
!ls pizza_steak/test/steak

import os
for dirpath,dirnames,filenames in os.walk("pizza_steak"):
    print(dirpath,dirnames,filenames)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255)
valid_datagen=ImageDataGenerator(rescale=1./255)

train_dir ="pizza_steak/train/"
test_dir ="pizza_steak/test/"

train_data = train_datagen.flow_from_directory(train_dir,
                                                batch_size=32,
                                                target_size=(224,224),
                                                class_mode="binary",
                                                seed=42)
valid_data = valid_datagen.flow_from_directory(test_dir,
                                                batch_size=32,
                                                target_size=(224,224),
                                                class_mode="binary",
                                                seed=42)

model_1=tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(224,224,3)),
    tf.keras.layers.Dense(4,activation="relu"),
    tf.keras.layers.Dense(4,activation="relu"),
    tf.keras.layers.Dense(1,activation="sigmoid")
])

model_1.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])


history_1=model_1.fit(train_data,
                epochs=8,
                steps_per_epoch=len(train_data),
                validation_data=valid_data,
                validation_steps=len(valid_data))

model_1.summary()

"""SECOND MODEL :with CNN"""

model_2=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=10,
                    kernel_size=3,
                    activation="relu",
                    input_shape=(224,224,3)),
    tf.keras.layers.Conv2D(10,3,activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2,
                        padding="valid"),
    tf.keras.layers.Conv2D(10,3,activation="relu"),
    tf.keras.layers.Conv2D(10,3,activation="relu"),
    tf.keras.layers.MaxPool2D(2),

    tf.keras.layers.Conv2D(10,3,activation="relu"),
    tf.keras.layers.MaxPool2D(2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1,activation="sigmoid")
])
model_2.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])


history_2=model_2.fit(train_data,
                                epochs=7,
                                steps_per_epoch=len(train_data),
                                validation_data=valid_data,
                                validation_steps=len(valid_data))

model_2.summary()

model_2.evaluate(valid_data)

valid_data.class_indices


y_predicted=model_2.predict(valid_data)

y_predicted[0]

valid_data[0]

import matplotlib.pyplot as plt

images, labels = valid_data[0]

plt.figure(figsize=(15,5))
for i in range(15):
    plt.subplot(1, 15, i+1)
    plt.imshow(images[i])
    plt.title(f"Label: {int(labels[i])}")
    plt.axis("off")
plt.show()

y_classes = (y_predicted > 0.5).astype(int)
print(y_classes[:15])