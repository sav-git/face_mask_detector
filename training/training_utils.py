import numpy as np
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