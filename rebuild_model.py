
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
import os

# Paths
BASE_DIR = r"D:\soft computing"
TRAIN_DIR = os.path.join(BASE_DIR, r"dataset\chest x ray\train")
VAL_DIR = os.path.join(BASE_DIR, r"dataset\chest x ray\val")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, r"models\DL\chest_xray_mobilenet_cpu.h5")

# Fast training setup to create a Keras 2 native model
img_size = (160, 160)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR, image_size=img_size, batch_size=32, label_mode='categorical'
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    VAL_DIR, image_size=img_size, batch_size=32, label_mode='categorical'
)

base_model = MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Very fast training - just 1 epoch to initialize weights and save as Keras 2
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=1)

model.save(MODEL_SAVE_PATH)
print("SUCCESS: Model saved in Keras 2 format.")
