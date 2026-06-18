import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from collections import Counter
import os

# ================= FORCE CPU =================
tf.config.set_visible_devices([], 'GPU')

# ================= PATH =================
BASE_DIR = r"D:\soft computing\dataset\eye"

TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR   = os.path.join(BASE_DIR, "validation")  # ✅ USE ACTUAL VALIDATION FOLDER
TEST_DIR  = os.path.join(BASE_DIR, "test")

MODEL_SAVE_PATH = r"D:\soft computing\models\DL\eye_mobilenet.h5"

# ================= PARAMETERS =================
IMG_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 25

# ================= DATA GENERATOR =================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.85, 1.15],
    shear_range=0.1
)

val_test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_test_gen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = val_test_gen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

NUM_CLASSES = train_data.num_classes
print("Classes:", train_data.class_indices)

# ================= CLASS WEIGHT (HANDLE IMBALANCE) =================
counter = Counter(train_data.classes)
total = sum(counter.values())
class_weight = {cls: total / (len(counter) * count) for cls, count in counter.items()}
print("Class weights:", class_weight)

# ================= MOBILENET MODEL =================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(160, 160, 3)
)

# Fine-tune last 30 layers for better accuracy
for layer in base_model.layers[:-30]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)

outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)

# ================= COMPILE =================
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ================= CALLBACKS =================
early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=6,
    restore_best_weights=True,
    mode="max"
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    patience=3,
    factor=0.3,
    min_lr=1e-7
)

checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

# ================= TRAIN =================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# ================= SAVE =================
model.save(MODEL_SAVE_PATH)
print("\n✅ Model saved at:", MODEL_SAVE_PATH)

# ================= TEST =================
test_loss, test_acc = model.evaluate(test_data)
print("\n🎯 Test Accuracy:", test_acc)