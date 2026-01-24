import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Input, AveragePooling2D, Dropout, 
    Flatten, Dense, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping,
    ReduceLROnPlateau, TensorBoard
)
import matplotlib.pyplot as plt
import numpy as np
import json
import os


class MaskDetectionModel:
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self, fine_tune_at=100):
        print("[INFO] Создание модели MobileNetV2...")
        
        base_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_tensor=Input(shape=self.input_shape)
        )
        
        base_model.trainable = False
        
        x = base_model.output
        
        x = AveragePooling2D(pool_size=(7, 7))(x)
        x = Flatten(name="flatten")(x)
        
        x = Dense(256, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        x = Dense(128, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        outputs = Dense(self.num_classes, activation="softmax")(x)
        
        self.model = Model(inputs=base_model.input, outputs=outputs)
        
        self.model.summary()
        
        return self.model
    
    def compile_model(self, learning_rate=1e-4):
        if self.model is None:
            self.build_model()
        
        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc")
            ]
        )
        
        print("[INFO] Модель скомпилирована")
        print(f"  Optimizer: Adam(lr={learning_rate})")
        print(f"  Loss: categorical_crossentropy")
        
        return self.model
    
    def create_callbacks(self, model_path="../models/mask_detector.model"):
        checkpoint = ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            verbose=1,
            save_best_only=True,
            mode="max"
        )
        
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=15,
            verbose=1,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        tensorboard = TensorBoard(
            log_dir="../logs",
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        
        return [checkpoint, early_stopping, reduce_lr, tensorboard]
    
    def unfreeze_layers(self, fine_tune_at=100):
        if self.model is None:
            raise ValueError("Модель не создана. Сначала вызовите build_model()")
        
        self.model.trainable = True
        
        for layer in self.model.layers[:fine_tune_at]:
            layer.trainable = False
        
        self.compile_model(learning_rate=1e-5)
        
        print(f"[INFO] Разморожено {len(self.model.layers) - fine_tune_at} слоев")
        print("[INFO] Начинаем fine-tuning...")

    def train(
        self, 
        trainX, trainY, 
        valX=None, valY=None,
        epochs=30,
        batch_size=32,
        use_augmentation=True
    ):
        if self.model is None:
            self.compile_model()
        
        print("[INFO] Начало обучения...")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Training samples: {len(trainX)}")
        if valX is not None:
            print(f"  Validation samples: {len(valX)}")
        
        callbacks = self.create_callbacks()
        
        if use_augmentation:
            from training_utils import create_augmentor
            datagen = create_augmentor()
            
            datagen.fit(trainX)
            
            history = self.model.fit(
                datagen.flow(trainX, trainY, batch_size=batch_size),
                steps_per_epoch=len(trainX) // batch_size,
                validation_data=(valX, valY) if valX is not None else None,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        else:
            history = self.model.fit(
                trainX, trainY,
                batch_size=batch_size,
                validation_data=(valX, valY) if valX is not None else None,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        
        self.model.save("../models/mask_detector_final.h5")
        self.save_training_history(history)
        self.plot_training_history(history)
        
        return history
    
    def save_training_history(self, history, path="../models/training_history.json"):
        history_dict = {}
        for key, values in history.history.items():
            history_dict[key] = [float(value) for value in values]
        
        with open(path, "w") as f:
            json.dump(history_dict, f, indent=4)
        
        print(f"[INFO] История обучения сохранена в {path}")
    
    def plot_training_history(self, history):
        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        metrics = [
            ('loss', 'val_loss', 'Loss'),
            ('accuracy', 'val_accuracy', 'Accuracy'),
            ('precision', 'val_precision', 'Precision'),
            ('recall', 'val_recall', 'Recall'),
            ('auc', 'val_auc', 'AUC')
        ]
        
        for idx, (train_metric, val_metric, title) in enumerate(metrics):
            if train_metric in history.history:
                ax = axes[idx // 3, idx % 3]
                
                ax.plot(history.history[train_metric], label=f'Train {title}')
                if val_metric in history.history:
                    ax.plot(history.history[val_metric], label=f'Val {title}')
                
                ax.set_title(f'{title} over Epochs')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(title)
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        for idx in range(len(metrics), 6):
            axes[idx // 3, idx % 3].axis('off')
        
        plt.tight_layout()
        plt.savefig("../models/training_metrics.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self, testX, testY):
        print("[INFO] Оценка модели на тестовых данных...")
        
        results = self.model.evaluate(testX, testY, verbose=0)
        
        metrics = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics[metric_name] = results[i]
            print(f"  {metric_name}: {results[i]:.4f}")
        
        predictions = self.model.predict(testX, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(testY, axis=1)
        
        from sklearn.metrics import classification_report, confusion_matrix
        import seaborn as sns
        
        print("\n[INFO] Classification Report:")
        print(classification_report(true_classes, predicted_classes, 
                                   target_names=['without_mask', 'with_mask']))
        
        cm = confusion_matrix(true_classes, predicted_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['without_mask', 'with_mask'],
                   yticklabels=['without_mask', 'with_mask'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig("../models/confusion_matrix.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        return metrics


def main():
    import joblib
    from training_utils import DatasetPreparer
    
    print("[=== ШАГ 1: ПОДГОТОВКА ДАННЫХ ===]")
    preparer = DatasetPreparer("../dataset")
    data, labels = preparer.load_images()
    data = preparer.normalize_data(data)
    encoded_labels = preparer.encode_labels(labels)
    
    trainX, valX, testX, trainY, valY, testY = preparer.split_dataset(
        data, encoded_labels, test_size=0.15, val_size=0.15
    )
    
    print("\n[=== ШАГ 2: СОЗДАНИЕ МОДЕЛИ ===]")
    model_builder = MaskDetectionModel()
    model_builder.build_model()
    model_builder.compile_model()
    
    print("\n[=== ШАГ 3: ОБУЧЕНИЕ МОДЕЛИ ===]")
    history = model_builder.train(
        trainX, trainY,
        valX, valY,
        epochs=25,
        batch_size=32,
        use_augmentation=True
    )
    
    print("\n[=== ШАГ 4: FINE-TUNING ===]")
    model_builder.unfreeze_layers(fine_tune_at=120)
    
    history_fine = model_builder.train(
        trainX, trainY,
        valX, valY,
        epochs=10,
        batch_size=16,
        use_augmentation=True
    )
    
    print("\n[=== ШАГ 5: ОЦЕНКА МОДЕЛИ ===]")
    metrics = model_builder.evaluate_model(testX, testY)
    
    print("\n[=== ШАГ 6: СОХРАНЕНИЕ ===]")
    model_builder.model.save("../models/mask_detector_final.h5")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model_builder.model)
    tflite_model = converter.convert()
    
    with open("../models/mask_detector.tflite", "wb") as f:
        f.write(tflite_model)
    
    print("[INFO] Обучение завершено успешно!")
    print(f"  Final accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"  Model saved: ../models/mask_detector_final.h5")
    print(f"  TFLite model saved: ../models/mask_detector.tflite")


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    
    main()