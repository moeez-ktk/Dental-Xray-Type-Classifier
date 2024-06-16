import cv2
import numpy as np
import tensorflow as tf

# Parameters
img_size = 224
categories = ['Cephalometric', 'Anteroposterior', 'OPG']

def prepare_image(image_path):
    img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_array = cv2.resize(img_array, (img_size, img_size))
    normalized_array = resized_array / 255.0
    return normalized_array.reshape(-1, img_size, img_size, 1)

def predict_image(image_path):
    model = tf.keras.models.load_model('best_model.keras')
    
    prepared_image = prepare_image(image_path)
    prediction = model.predict(prepared_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    print(f'Predicted class: {categories[predicted_class]}')

if __name__ == "__main__":
    image_path = 'path_to_your_image.png'
    predict_image(image_path)
