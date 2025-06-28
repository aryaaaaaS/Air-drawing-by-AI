import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf

class ShapeModelTrainer:
    def __init__(self):
        self.dataset_path = "dataset"
        self.model_save_path = "models/drawing_model.h5"
        self.label_encoder_path = "models/drawing_model_label_encoder.npy"

        # Dynamically load shape classes from dataset folders
        self.shape_classes = sorted([
            folder for folder in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, folder))
        ])
        print(f"Detected shape classes: {self.shape_classes}")

        self.img_size = (64, 64)
        self.x_data = []
        self.y_data = []

    def load_images(self):
        print("Loading images...")
        for label in self.shape_classes:
            path = os.path.join(self.dataset_path, label)
            for file in os.listdir(path):
                img_path = os.path.join(path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, self.img_size)
                self.x_data.append(img)
                self.y_data.append(label)

        self.x_data = np.array(self.x_data).reshape(-1, 64, 64, 1) / 255.0
        self.y_data = np.array(self.y_data)

        print(f"Total images: {len(self.x_data)}")

    def preprocess_labels(self):
        print("Encoding labels...")
        self.encoder = LabelEncoder()
        self.y_encoded = self.encoder.fit_transform(self.y_data)
        self.y_categorical = to_categorical(self.y_encoded)

        # Save encoder classes
        np.save(self.label_encoder_path, self.encoder.classes_)
        print(f"Labels encoded and saved to: {self.label_encoder_path}")

    def build_model(self):
        print("Building model...")
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(self.shape_classes), activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def train_model(self, model):
        print("Splitting data and augmenting...")

        x_train, x_test, y_train, y_test = train_test_split(
            self.x_data, self.y_categorical, test_size=0.2, random_state=42)

        datagen = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1
        )
        datagen.fit(x_train)

        print("Training model...")
        history = model.fit(datagen.flow(x_train, y_train, batch_size=32),
                            validation_data=(x_test, y_test),
                            epochs=25,
                            verbose=1)

        print("Saving model...")
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        model.save(self.model_save_path)
        print(f"Model saved to {self.model_save_path}")

        self.plot_results(history, x_test, y_test, model)

    def plot_results(self, history, x_test, y_test, model):
        print("Plotting training results...")

        # Accuracy/Loss
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Confusion matrix
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)
        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap="Blues",
                    xticklabels=self.encoder.classes_,
                    yticklabels=self.encoder.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

def main():
    trainer = ShapeModelTrainer()
    trainer.load_images()
    trainer.preprocess_labels()
    model = trainer.build_model()
    trainer.train_model(model)

if __name__ == "__main__":
    main()
