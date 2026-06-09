import os
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Large, ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        print("GPU terdeteksi:")
        for gpu in gpus:
            print(gpu)
    else:
        print("GPU tidak terdeteksi. Training memakai CPU.")


def find_dataset_root(start_dir):
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    start_dir = Path(start_dir)

    for root, dirs, files in os.walk(start_dir):
        root_path = Path(root)
        valid_folders = []

        for folder in dirs:
            folder_path = root_path / folder
            image_count = 0

            for file in folder_path.iterdir():
                if file.is_file() and file.suffix.lower() in image_extensions:
                    image_count += 1

            if image_count > 0:
                valid_folders.append(folder)

        if len(valid_folders) >= 2:
            return str(root_path)

    return None


def collect_dataset(dataset_dir, limit_per_class=None):
    dataset_dir = Path(dataset_dir)
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    class_names = sorted([
        folder.name for folder in dataset_dir.iterdir()
        if folder.is_dir()
    ])

    data = []

    for label_index, class_name in enumerate(class_names):
        class_dir = dataset_dir / class_name

        image_paths = [
            str(file) for file in class_dir.iterdir()
            if file.is_file() and file.suffix.lower() in image_extensions
        ]

        image_paths = sorted(image_paths)

        if limit_per_class is not None:
            image_paths = image_paths[:limit_per_class]

        for image_path in image_paths:
            data.append({
                "image_path": image_path,
                "label": label_index,
                "class_name": class_name
            })

    df = pd.DataFrame(data)
    return df, class_names


def load_image(image_path, label, image_size):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32)

    return image, label


def make_dataset(df, image_size, batch_size, shuffle=False):
    paths = df["image_path"].values
    labels = df["label"].values.astype(np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df), seed=42)

    dataset = dataset.map(
        lambda path, label: load_image(path, label, image_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def split_dataset(df, seed):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df["label"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["label"]
    )

    return train_df, val_df, test_df


def build_augmentation():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10)
    ], name="augmentation")


def build_mobilenetv3(num_classes, image_size):
    inputs = layers.Input(shape=(image_size[0], image_size[1], 3))

    augmentation = build_augmentation()

    base_model = MobileNetV3Large(
        input_shape=(image_size[0], image_size[1], 3),
        include_top=False,
        weights="imagenet",
        include_preprocessing=True
    )

    base_model.trainable = False

    x = augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="MobileNetV3Large")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def build_resnet50(num_classes, image_size):
    inputs = layers.Input(shape=(image_size[0], image_size[1], 3))

    augmentation = build_augmentation()

    base_model = ResNet50(
        input_shape=(image_size[0], image_size[1], 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    x = augmentation(inputs)
    x = resnet_preprocess(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="ResNet50")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def get_callbacks(model_name):
    os.makedirs("models", exist_ok=True)

    return [
        EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=f"models/{model_name}_best.keras",
            monitor="val_accuracy",
            save_best_only=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7
        )
    ]


def plot_history(history, model_name):
    os.makedirs("results", exist_ok=True)

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.title(f"{model_name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/{model_name}_accuracy.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.title(f"{model_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/{model_name}_loss.png", dpi=300)
    plt.close()


def get_predictions(model, dataset):
    y_true = []
    y_pred = []

    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)

        y_true.extend(labels.numpy())
        y_pred.extend(predicted_labels)

    return np.array(y_true), np.array(y_pred)


def save_confusion_matrix(y_true, y_pred, class_names, model_name):
    os.makedirs("results", exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix {model_name}")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                horizontalalignment="center",
                verticalalignment="center"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"results/{model_name}_confusion_matrix.png", dpi=300)
    plt.close()

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(f"results/{model_name}_confusion_matrix.csv")


def evaluate_model(model, test_ds, class_names, model_name):
    os.makedirs("results", exist_ok=True)

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

    y_true, y_pred = get_predictions(model, test_ds)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )

    print(f"\nClassification Report {model_name}")
    print(report_text)

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(f"results/{model_name}_classification_report.csv")

    with open(f"results/{model_name}_classification_report.txt", "w") as file:
        file.write(report_text)

    save_confusion_matrix(y_true, y_pred, class_names, model_name)

    summary = {
        "model": model_name,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "macro_precision": report_dict["macro avg"]["precision"],
        "macro_recall": report_dict["macro avg"]["recall"],
        "macro_f1_score": report_dict["macro avg"]["f1-score"],
        "weighted_precision": report_dict["weighted avg"]["precision"],
        "weighted_recall": report_dict["weighted avg"]["recall"],
        "weighted_f1_score": report_dict["weighted avg"]["f1-score"]
    }

    return summary


def train_model(model_name, model, train_ds, val_ds, test_ds, class_names, epochs):
    print("\n" + "=" * 70)
    print(f"Training model: {model_name}")
    print("=" * 70)

    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=get_callbacks(model_name)
    )

    model.save(f"models/{model_name}_final.keras")

    plot_history(history, model_name)

    summary = evaluate_model(
        model=model,
        test_ds=test_ds,
        class_names=class_names,
        model_name=model_name
    )

    return summary


def predict_single_image(model_path, image_path, class_names_path, image_size):
    class_names = pd.read_csv(class_names_path)["class_name"].tolist()

    model = tf.keras.models.load_model(model_path)

    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, axis=0)

    prediction = model.predict(image)
    predicted_index = int(np.argmax(prediction, axis=1)[0])
    confidence = float(np.max(prediction))

    print("Gambar:", image_path)
    print("Prediksi:", class_names[predicted_index])
    print("Confidence:", confidence)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="data/gameplay-images")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--only", type=str, default="all", choices=["all", "mobilenetv3", "resnet50"])

    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--class-names-path", type=str, default="results/class_names.csv")

    args = parser.parse_args()

    image_size = (args.img_size, args.img_size)

    set_seed(args.seed)
    setup_gpu()

    if args.predict:
        if args.model_path is None or args.image_path is None:
            print("Untuk prediksi, isi --model-path dan --image-path.")
            return

        predict_single_image(
            model_path=args.model_path,
            image_path=args.image_path,
            class_names_path=args.class_names_path,
            image_size=image_size
        )
        return

    dataset_root = find_dataset_root(args.dataset)

    if dataset_root is None:
        print("Folder dataset tidak ditemukan.")
        print("Pastikan folder dataset berisi subfolder kelas seperti Minecraft, Roblox, dan lainnya.")
        return

    print("Dataset root:", dataset_root)

    df, class_names = collect_dataset(
        dataset_dir=dataset_root,
        limit_per_class=args.limit_per_class
    )

    if len(df) == 0:
        print("Dataset kosong.")
        return

    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    pd.DataFrame({"class_name": class_names}).to_csv(
        "results/class_names.csv",
        index=False
    )

    print("\nJumlah data:", len(df))
    print("Jumlah kelas:", len(class_names))
    print("\nDistribusi data:")
    print(df["class_name"].value_counts())

    train_df, val_df, test_df = split_dataset(df, args.seed)

    print("\nJumlah train:", len(train_df))
    print("Jumlah validation:", len(val_df))
    print("Jumlah test:", len(test_df))

    train_df.to_csv("results/train_split.csv", index=False)
    val_df.to_csv("results/validation_split.csv", index=False)
    test_df.to_csv("results/test_split.csv", index=False)

    train_ds = make_dataset(
        train_df,
        image_size=image_size,
        batch_size=args.batch,
        shuffle=True
    )

    val_ds = make_dataset(
        val_df,
        image_size=image_size,
        batch_size=args.batch,
        shuffle=False
    )

    test_ds = make_dataset(
        test_df,
        image_size=image_size,
        batch_size=args.batch,
        shuffle=False
    )

    summaries = []

    if args.only in ["all", "mobilenetv3"]:
        mobilenet_model = build_mobilenetv3(
            num_classes=len(class_names),
            image_size=image_size
        )

        mobilenet_summary = train_model(
            model_name="MobileNetV3Large",
            model=mobilenet_model,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            class_names=class_names,
            epochs=args.epochs
        )

        summaries.append(mobilenet_summary)

    if args.only in ["all", "resnet50"]:
        resnet_model = build_resnet50(
            num_classes=len(class_names),
            image_size=image_size
        )

        resnet_summary = train_model(
            model_name="ResNet50",
            model=resnet_model,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            class_names=class_names,
            epochs=args.epochs
        )

        summaries.append(resnet_summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv("results/final_comparison.csv", index=False)

    print("\nHasil akhir:")
    print(summary_df)

    print("\nSelesai.")
    print("Model tersimpan di folder models/")
    print("Grafik, confusion matrix, dan report tersimpan di folder results/")


if __name__ == "__main__":
    main()