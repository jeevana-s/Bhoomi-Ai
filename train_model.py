import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

IMG_SIZE = 224
BATCH = 32

# Verify dataset exists
if not os.path.exists("dataset"):
    print("❌ Error: 'dataset' folder not found in current directory.")
    exit(1)

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

print("Loading training data...")
train_data = datagen.flow_from_directory(
    "dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical",
    subset="training"
)

print("Loading validation data...")
val_data = datagen.flow_from_directory(
    "dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical",
    subset="validation"
)

print(f"Classes found: {train_data.class_indices}")

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(IMG_SIZE,IMG_SIZE,3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(train_data.num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("Starting training...")
model.fit(train_data, validation_data=val_data, epochs=5)

# ⚠️ VERY IMPORTANT
# Save class order inside model
# model.class_indices = train_data.class_indices 
# (Note: Standard Keras SavedModel doesn't persist arbitrary attributes easily, 
# but the folder structure implies class order 0,1,2,3... alphabetically)

print("Saving model to 'model_saved'...")
model.export("model_saved")

print("✅ Model trained and saved successfully")
