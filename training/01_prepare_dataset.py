import os
import cv2
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import joblib


class DatasetPreparer:  
      
    def __init__(self, dataset_path, output_size=(224, 224)):
        self.dataset_path = dataset_path
        self.output_size = output_size
        self.label_encoder = LabelEncoder()
        
    def load_images(self):
        print("[INFO] Загрузка изображений из датасета...")
        
        image_paths = list(paths.list_images(self.dataset_path))
        
        if not image_paths:
            raise ValueError(f"Не найдены изображения в {self.dataset_path}")
        
        data = []
        labels = []
        
        for image_path in image_paths:
            label = image_path.split(os.path.sep)[-2]
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"[WARNING] Не удалось загрузить: {image_path}")
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            image = cv2.resize(image, self.output_size)
            
            data.append(image)
            labels.append(label)
        
        data = np.array(data, dtype="float32")
        labels = np.array(labels)
        
        print(f"[INFO] Загружено {len(data)} изображений")
        print(f"[INFO] Распределение классов: {np.unique(labels, return_counts=True)}")
        
        return data, labels
    
    def normalize_data(self, data):
        return data / 255.0
    
    def encode_labels(self, labels):
        integer_labels = self.label_encoder.fit_transform(labels)
        
        num_classes = len(self.label_encoder.classes_)
        one_hot_labels = np.eye(num_classes)[integer_labels]
        
        print(f"[INFO] Классы: {self.label_encoder.classes_}")
        
        return one_hot_labels
    
    def split_dataset(self, data, labels, test_size=0.2, val_size=0.1):
        trainX, testX, trainY, testY = train_test_split(
            data, labels, 
            test_size=test_size, 
            stratify=labels,
            random_state=42
        )
        
        if val_size > 0:
            trainX, valX, trainY, valY = train_test_split(
                trainX, trainY,
                test_size=val_size,
                stratify=trainY,
                random_state=42
            )
        else:
            valX, valY = None, None
        
        print(f"[INFO] Размеры данных:")
        print(f"  Train: {trainX.shape[0]} изображений")
        if valX is not None:
            print(f"  Validation: {valX.shape[0]} изображений")
        print(f"  Test: {testX.shape[0]} изображений")
        
        return trainX, valX, testX, trainY, valY, testY
    
    def create_augmentor(self):
        return ImageDataGenerator(
            rotation_range=15,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode="nearest"
        )


def main():
    preparer = DatasetPreparer("../dataset")
    
    data, labels = preparer.load_images()
    
    data = preparer.normalize_data(data)
    
    encoded_labels = preparer.encode_labels(labels)
    
    trainX, valX, testX, trainY, valY, testY = preparer.split_dataset(
        data, encoded_labels
    )
    
    joblib.dump(preparer.label_encoder, "../models/label_encoder.pkl")
    
    return trainX, valX, testX, trainY, valY, testY


if __name__ == "__main__":
    trainX, valX, testX, trainY, valY, testY = main()
    print(f"\n[INFO] Подготовка датасета завершена успешно!")
    print(f"  Train data shape: {trainX.shape}")
    print(f"  Train labels shape: {trainY.shape}")