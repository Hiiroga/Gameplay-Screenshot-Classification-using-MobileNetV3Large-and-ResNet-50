"""
Script to extract the missing MobileNetV3Large test loss.
Loads the saved model and evaluates it on the test set.
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from pathlib import Path

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

def load_image(image_path, label, image_size=(224, 224)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32)
    return image, label

def main():
    set_seed(42)
    setup_gpu()

    # Load the test split
    test_df = pd.read_csv("results/test_split.csv")
    print(f"Test samples: {len(test_df)}")

    paths = test_df["image_path"].values
    labels = test_df["label"].values.astype(np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(
        lambda p, l: load_image(p, l, (224, 224)),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(16).prefetch(tf.data.AUTOTUNE)

    # Load model
    model_path = "models/MobileNetV3Large_best.keras"
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Evaluate
    test_loss, test_accuracy = model.evaluate(dataset, verbose=1)
    print(f"\n{'='*50}")
    print(f"MobileNetV3Large Test Loss:     {test_loss:.10f}")
    print(f"MobileNetV3Large Test Accuracy: {test_accuracy:.10f}")
    print(f"{'='*50}")

    # Save result
    result = {
        "model": "MobileNetV3Large",
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    }
    pd.DataFrame([result]).to_csv("results/mobilenet_test_loss.csv", index=False)
    print("Result saved to results/mobilenet_test_loss.csv")

if __name__ == "__main__":
    main()
