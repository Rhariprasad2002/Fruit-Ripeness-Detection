import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import json

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

train_dir = "dataset/Train"
test_dir  = "dataset/Test"

# ---------------------------------------------------
# IMAGE GENERATORS
# ---------------------------------------------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# ---------------------------------------------------
# SAVE CLASS ORDER (VERY IMPORTANT)
# ---------------------------------------------------
class_indices = train_data.class_indices
classes = sorted(class_indices, key=class_indices.get)

with open("labels.json","w") as f:
    json.dump(classes, f)

print("Class Order:", classes)

# ---------------------------------------------------
# HANDLE CLASS IMBALANCE  ⭐⭐⭐
# ---------------------------------------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# ---------------------------------------------------
# TRANSFER LEARNING MODEL (MobileNetV2)
# ---------------------------------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False  # freeze pretrained layers

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(len(classes), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------------------------------------------
# CALLBACKS (SAVE BEST MODEL)
# ---------------------------------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "fruit_model.h5",
    monitor="val_accuracy",
    save_best_only=True
)

# ---------------------------------------------------
# TRAIN MODEL  ⭐⭐⭐
# ---------------------------------------------------
model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=test_data,
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint]
)

print("\n🎉 TRAINING COMPLETE")
print("Best model saved as fruit_model.h5")
