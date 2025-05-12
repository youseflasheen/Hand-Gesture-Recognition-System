import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import re
import cv2
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for skin detection in augmentation
lower_skin = np.array([0, 10, 60], dtype=np.uint8)
upper_skin = np.array([25, 180, 255], dtype=np.uint8)

def data_generator(filenames, data_dir, batch_size, img_height, img_width, class_to_idx, class_names, shuffle=True):
    """Generate batches of images and labels."""
    if not filenames:
        raise ValueError(f"No valid images provided to generator")
    if shuffle:
        np.random.shuffle(filenames)

    i = 0
    while True:
        batch_images = []
        batch_labels = []
        attempts = 0
        max_attempts = len(filenames) * 2
        while len(batch_images) < batch_size and attempts < max_attempts:
            if i >= len(filenames):
                i = 0
                if shuffle:
                    np.random.shuffle(filenames)
                logger.info("Resetting data generator index and shuffling filenames.")
            filename = filenames[i]
            img_path = os.path.join(data_dir, filename)
            try:
                img = load_img(img_path, target_size=(img_height, img_width))
                img_array = img_to_array(img) / 255.0
                if img_array is None or np.isnan(img_array).any() or np.isinf(img_array).any():
                    logger.warning(f"Invalid image array: {img_path}")
                    i += 1
                    attempts += 1
                    continue
            except Exception as e:
                logger.warning(f"Failed to load image: {img_path} ({e})")
                i += 1
                attempts += 1
                continue
            match = re.match(r'(.+)_processed_', filename)
            if not match or match.group(1) not in class_to_idx:
                logger.warning(f"Invalid class in filename: {filename}")
                i += 1
                attempts += 1
                continue
            label_idx = class_to_idx[match.group(1)]
            batch_images.append(img_array)
            batch_labels.append(label_idx)
            i += 1
            attempts += 1

        if not batch_images:
            logger.error("No valid images could be loaded for this batch after max attempts.")
            raise StopIteration

        batch_images = np.array(batch_images)
        batch_labels = tf.keras.utils.to_categorical(batch_labels, num_classes=len(class_names))
        yield batch_images, batch_labels

def add_random_background(image):
    """Add random background to the image."""
    logger.debug(f"Input image dtype: {image.dtype}, shape: {image.shape}")

    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[-1] == 3 else image
    background = np.random.rand(*image.shape).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    hand = cv2.bitwise_and(image, image, mask=mask)
    inv_mask = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(background, background, mask=inv_mask)

    logger.debug(f"Hand dtype: {hand.dtype}, shape: {hand.shape}")
    logger.debug(f"Background dtype: {bg.dtype}, shape: {bg.shape}")

    result = cv2.add(hand, bg)

    if np.isnan(result).any() or np.isinf(result).any():
        logger.error("NaN or Inf detected in augmentation result.")
        raise ValueError("Invalid augmentation output: NaN or Inf values.")
    if result.min() < 0 or result.max() > 1:
        logger.warning(f"Augmentation result out of range: min={result.min()}, max={result.max()}. Clipping to [0, 1].")
        result = np.clip(result, 0, 1)

    logger.debug(f"Result dtype: {result.dtype}, shape: {result.shape}")
    return result

def create_data_generator(filenames, data_dir, batch_size, img_height, img_width, class_to_idx, class_names, shuffle=True, augment=True):
    """Create data generator with optional background augmentation."""
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        preprocessing_function=add_random_background if augment else None
    )
    def wrapped_generator():
        base_gen = data_generator(filenames, data_dir, batch_size, img_height, img_width, class_to_idx, class_names, shuffle)
        skipped_batches = 0
        while True:
            try:
                batch_images, batch_labels = next(base_gen)
                augmented_batch = next(datagen.flow(batch_images, batch_labels, batch_size=batch_size, shuffle=False))
                # Save a few augmented images for debugging
                if skipped_batches == 0 and augment:
                    for j in range(min(3, len(batch_images))):
                        plt.imsave(f"debug_augmented_{j}.png", augmented_batch[0][j])
                yield augmented_batch
            except Exception as e:
                logger.warning(f"Augmentation failed: {e}")
                skipped_batches += 1
                if skipped_batches > 100:
                    logger.error("Too many augmentation failures. Stopping generator.")
                    raise StopIteration
                continue
    return wrapped_generator()

def evaluate_model(model, data_dir, batch_size, img_height, img_width, class_to_idx, class_names, num_samples=5000):
    val_gen = data_generator(data_dir, batch_size, img_height, img_width, class_to_idx, class_names, shuffle=False)
    y_true, y_pred = [], []
    
    samples_processed = 0
    for batch_images, batch_labels in val_gen:
        batch_pred = model.predict(batch_images, verbose=0)
        y_true.extend(np.argmax(batch_labels, axis=1))
        y_pred.extend(np.argmax(batch_pred, axis=1))
        samples_processed += len(batch_images)
        if samples_processed >= num_samples:
            break
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation F1-Score: {f1:.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_training_history(history, fine_tune_history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    train_acc = history.history['accuracy'] + fine_tune_history.history['accuracy']
    val_acc = history.history['val_accuracy'] + fine_tune_history.history['val_accuracy']
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    train_loss = history.history['loss'] + fine_tune_history.history['loss']
    val_loss = history.history['val_loss'] + fine_tune_history.history['val_loss']
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.savefig('training_plots.png')
    plt.close()

def main():
    # Set paths
    data_dir = r"E:\new yousef\hand-gesture-recognition\data\processed"
    model_save_path = r"E:\new yousef\hand-gesture-recognition\models\asl_gesture_recognition_model.h5"
    img_height = 96
    img_width = 96
    batch_size = 32
    epochs = 20
    num_classes = 5

    # Define class names
    class_names = ['hello', 'yes', 'no', 'i love you', 'thank you']
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    # Load and split filenames
    valid_filenames = [f for f in os.listdir(data_dir) if f.endswith('.png') and re.match(r'(.+)_processed_', f) and re.match(r'(.+)_processed_', f).group(1) in class_names]
    total_images = len(valid_filenames)
    print(f"Total valid images: {total_images}")
    if total_images == 0:
        print(f"No valid images found in {data_dir}")
        return

    # Split into train and validation
    train_filenames, val_filenames = train_test_split(valid_filenames, train_size=0.8, random_state=42, shuffle=True)
    print(f"Training images: {len(train_filenames)}, Validation images: {len(val_filenames)}")

    # Log label distribution
    train_labels = [re.match(r'(.+)_processed_', f).group(1) for f in train_filenames]
    val_labels = [re.match(r'(.+)_processed_', f).group(1) for f in val_filenames]
    logger.info(f"Training label distribution: {np.unique(train_labels, return_counts=True)}")
    logger.info(f"Validation label distribution: {np.unique(val_labels, return_counts=True)}")

    # Define steps per epoch and validation steps
    steps_per_epoch = max(1, len(train_filenames) // batch_size)
    validation_steps = max(1, len(val_filenames) // batch_size)

    # Create generators
    train_generator = create_data_generator(
        train_filenames, data_dir, batch_size, img_height, img_width, 
        class_to_idx, class_names, shuffle=True, augment=False  # Disable custom augmentation for now
    )
    val_generator = create_data_generator(
        val_filenames, data_dir, batch_size, img_height, img_width, 
        class_to_idx, class_names, shuffle=False, augment=False
    )

    # Define the model using MobileNetV2
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Reduced dropout
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model (initial training)
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs // 2,
        validation_data=val_generator,
        validation_steps=validation_steps,
        verbose=1
    )

    # Fine-tune the model
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    fine_tune_history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs // 2,
        validation_data=val_generator,
        validation_steps=validation_steps,
        verbose=1
    )

    # Save the model
    try:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")
    except PermissionError:
        print(f"Error: No permission to save model to '{model_save_path}'. Try running as administrator or checking folder permissions.")
        return
    except OSError as e:
        print(f"Error: Failed to save model to '{model_save_path}'. Reason: {e}")
        return

    # Evaluate and plot results
    evaluate_model(model, val_filenames, batch_size, img_height, img_width, class_to_idx, class_names)
    plot_training_history(history, fine_tune_history)

if __name__ == "__main__":
    main()