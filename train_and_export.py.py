import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# 1. Configuration & Hyperparameters
IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE = 32

# 2. Dataset Paths (Kaggle Environment)
DATA_DIR = '/kaggle/input/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product'
TRAIN_DIR = os.path.join(DATA_DIR, 'casting_data/casting_data/train')
TEST_DIR = os.path.join(DATA_DIR, 'casting_data/casting_data/test')

print("✅ Configuration initialized successfully!")

# ==========================================
# 1. Load and Split Dataset
# ==========================================
raw_train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

raw_val_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

raw_test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

class_names = raw_train_ds.class_names
print(f"Detected Classes: {class_names}") # ['def_front', 'ok_front']

# Data Pipeline Optimization
def prepare_pipeline(ds, is_training=False):
    if is_training:
        ds = ds.cache().shuffle(buffer_size=1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    else:
        ds = ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

train_ds = prepare_pipeline(raw_train_ds, is_training=True)
val_ds = prepare_pipeline(raw_val_ds, is_training=False)
test_ds = prepare_pipeline(raw_test_ds, is_training=False)

# ==========================================
# 2. Build MobileNetV3 Architecture (Softmax)
# ==========================================

# Data Augmentation Layer
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.1),
    layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
], name="industrial_data_augmentation")

# Built-in Preprocessing Layer
preprocess_layer = keras.applications.mobilenet_v3.preprocess_input

# Load Base Model
base_model = keras.applications.MobileNetV3Small(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False # Initial freezing

# Functional Model Construction
inputs = keras.Input(shape=IMG_SHAPE, name="input_image")
x = data_augmentation(inputs)
x = preprocess_layer(x)  # Integrated color preprocessing
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D(name="global_avg_pooling")(x)
x = layers.BatchNormalization(name="batch_norm")(x)
x = layers.Dense(128, activation='relu', name="dense_features")(x)
x = layers.Dropout(0.3, name="dropout_layer")(x)

# 2 Outputs with Softmax Activation (Matches Azure Standard)
outputs = layers.Dense(len(class_names), activation='softmax', name="output_probabilities")(x)

industrial_model = keras.Model(inputs, outputs, name="Casting_Quality_Inspector_MobileNetV3")

industrial_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',  # Compatible with Softmax
    metrics=['accuracy']
)

industrial_model.summary()

# ==========================================
# 3. Model Training & Fine-Tuning
# ==========================================
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6),
    keras.callbacks.ModelCheckpoint('best_casting_mobilenetv3.keras', monitor='val_loss', save_best_only=True)
]

print("\n--- Phase 1: Training Classification Head ---")
history_warmup = industrial_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8,
    callbacks=callbacks
)

# Fine-Tuning Stage
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 20
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

industrial_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n--- Phase 2: Fine-Tuning Model ---")
history_finetune = industrial_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=6,
    callbacks=callbacks
)

# ==========================================
# 4. ONNX Export
# ==========================================
!pip install tf2onnx -q
import tf2onnx
import onnx

input_signature = [tf.TensorSpec(shape=(None, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32, name="input_image")]
onnx_model, _ = tf2onnx.convert.from_keras(industrial_model, input_signature=input_signature, opset=13)

onnx_path = "casting_model.onnx"
onnx.save(onnx_model, onnx_path)

print(f"\n✅ ONNX model successfully exported to: {onnx_path}")
print(f"Class mapping order: {class_names}")

# ==========================================
# 5. Plot Performance Metrics
# ==========================================
def plot_industrial_history(history_warmup, history_finetune):
    acc = history_warmup.history['accuracy'] + history_finetune.history['accuracy']
    val_acc = history_warmup.history['val_accuracy'] + history_finetune.history['val_accuracy']
    loss = history_warmup.history['loss'] + history_finetune.history['loss']
    val_loss = history_warmup.history['val_loss'] + history_finetune.history['val_loss']
    warmup_epochs = len(history_warmup.history['accuracy'])

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy', color='#1f77b4', linewidth=2.5)
    plt.plot(val_acc, label='Validation Accuracy', color='#ff7f0e', linewidth=2.5)
    plt.axvline(x=warmup_epochs-1, color='green', linestyle='--', label='Start Fine-Tuning', linewidth=2)
    plt.title('MobileNetV3 Accuracy Curve', fontsize=13, fontweight='bold')
    plt.xlabel('Epochs', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss', color='#1f77b4', linewidth=2.5)
    plt.plot(val_loss, label='Validation Loss', color='#ff7f0e', linewidth=2.5)
    plt.axvline(x=warmup_epochs-1, color='green', linestyle='--', label='Start Fine-Tuning', linewidth=2)
    plt.title('MobileNetV3 Loss Curve', fontsize=13, fontweight='bold')
    plt.xlabel('Epochs', fontsize=11)
    plt.ylabel('Loss', fontsize=11)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()

plot_industrial_history(history_warmup, history_finetune)