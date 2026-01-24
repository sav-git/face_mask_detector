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