import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Paths
data_dir = './dataset'
categories = ['Cephalometric', 'Anteroposterior', 'OPG']

# Parameters
img_size = 224  # Resize images to 224x224

def load_data(data_dir, categories, img_size):
    data = []
    labels = []
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_array = cv2.resize(img_array, (img_size, img_size))
                data.append(resized_array)
                labels.append(class_num)
            except Exception as e:
                pass
    data = np.array(data).reshape(-1, img_size, img_size, 1)
    data = data / 255.0  # Normalize pixel values
    return data, np.array(labels)

def prepare_data():
    data, labels = load_data(data_dir, categories, img_size)
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
    np.save('X_train.npy', X_train)
    np.save('X_val.npy', X_val)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)

if __name__ == "__main__":
    prepare_data()
