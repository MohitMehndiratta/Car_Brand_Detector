from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
import numpy as np
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import os
import tensorflow as tf



# app = Flask(__name__)

model_path = r'C:\Users\Mohit\PycharmProjects\ML_Sample_codes_Repository\car_model_resnet50.h5'
my_model = load_model(model_path)


def model_predict(image_path, model):
    global graph
    graph = tf.compat.v1.get_default_graph

    with graph.as_default():
        img = load_img(image_path, target_size=(224, 224))
        x = img_to_array(img)
        x = x / 255
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)


        predctn = model.predict(x)
        predctn = np.argmax(predctn, axis=1)

        print(predctn)
        if predctn == 0:
            predctn = 'This Car is Audi'
        elif predctn == 1:
            predctn = 'This car is Lamborghini'
        else:
            predctn = 'This car is Mercedes'

    return predctn

# @app.route('/', methods=['GET'])
# def index():
#     # Main page
#     return render_template('index.html')


# @app.route('/predict', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         # Get the file from post request
#         f = request.files['file']
#
#         # Save the file to ./uploads
#         basepath = os.path.dirname(__file__)
#         file_path = os.path.join(
#             basepath, 'uploads', secure_filename(f.filename))
#         f.save(file_path)
#
#         # Make prediction
#         car_preds = model_predict(r'C:\Users\Mohit\Desktop\Sample Files\Car Dataset\Datasets\Test\lamborghini\4.jpg', my_model)
#         result = car_preds
#         return result
#     return None


# if __name__ == '__main__':
#     app.run(debug=True)

# Test the Model----
print(model_predict(r'C:\Users\Mohit\Desktop\Sample Files\Car Dataset\Datasets\Test\lamborghini\4.jpg', my_model))