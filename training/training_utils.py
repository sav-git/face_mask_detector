import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_augmentor():
    return ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.5, 1.5],
        channel_shift_range=20.0,
        fill_mode='nearest'
    )

def create_augmentor_fast():
    """Упрощенная аугментация для быстрого обучения"""
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

def load_split_data(split_path, target_size=(224,224)):
    """
    split_path: путь к папке типа train/ или val/ , внутри которой папки классов
    возвращает (images, labels)
    """
    data = []
    labels = []
    class_names = sorted(os.listdir(split_path))
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    
    for class_name in class_names:
        class_dir = os.path.join(split_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.png','.jpg','.jpeg')):
                path = os.path.join(class_dir, fname)
                img = cv2.imread(path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, target_size)
                data.append(img)
                labels.append(class_name)
    
    data = np.array(data, dtype='float32') / 255.0
    int_labels = label_encoder.transform(labels)
    one_hot = to_categorical(int_labels, num_classes=len(class_names))
    return data, one_hot, label_encoder