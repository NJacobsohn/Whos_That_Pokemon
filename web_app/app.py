# Demonstrates Bootstrap version 3.3 Starter Template
# available here: https://getbootstrap.com/docs/3.3/getting-started/#examples

from flask import Flask, render_template, flash, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from PIL import Image, ExifTags
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
import pickle

UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def rotate_save(f, file_path):
    try:
        image=Image.open(f)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        exif=dict(image._getexif().items())

        if exif[orientation] == 3:
            image=image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image=image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image=image.rotate(90, expand=True)
        image.save(file_path)
        image.close()

    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        image.save(file_path)
        image.close()

def process_img(filename):
    original = load_img(filename, target_size=(64, 64))
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)

    return image_batch

def model_predict(img_path, model):
    im =  process_img(img_path)
    preds =  model.predict(im)
    top_3 = preds.argsort()[0][::-1][:3] # sort in reverse order and return top 3 indices
    top_3_names = class_names[top_3]
    top_3_percent = preds[0][[top_3]]*100
    top_3_text = '<br>'.join([f'{name}: {percent:.2f}%' for name, percent in zip(top_3_names,top_3_percent)])
    return top_3_text


# home page
@app.route('/')
def index():
    return render_template('index.html')

# about page
@app.route('/about/')
def about():
    return render_template('about.html')

# contact page
@app.route('/contact/')
def contact():
    return render_template('contact.html')

# prediction page
@app.route('/predict/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        
        f = request.files['file']


        
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        # f.save(file_path)
        rotate_save(f, file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Delete it so we don't clutter our server up
        os.remove(file_path)

        return preds
    return None

if __name__ == '__main__':

    model = load_model("../models/model_acc2713.h5")
    with open('../pickles/class_names.p', 'rb') as f:
                class_names = np.array(pickle.load(f))
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=False)
