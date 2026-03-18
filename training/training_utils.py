import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_augmentor():
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.7, 1.3],
        channel_shift_range=20.0,
        fill_mode='nearest'
    )


def calculate_class_weights(labels):
    from sklearn.utils.class_weight import compute_class_weight
    
    integer_labels = np.argmax(labels, axis=1)
    
    classes = np.unique(integer_labels)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=integer_labels
    )
    
    return dict(zip(classes, weights))


def learning_rate_schedule(epoch, lr):
    if epoch < 10:
        return lr
    elif epoch < 20:
        return lr * 0.5
    elif epoch < 30:
        return lr * 0.2
    else:
        return lr * 0.1

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