from flask import Flask, render_template, flash, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from PIL import Image, ExifTags
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
import pickle

UPLOAD_FOLDER = '/uploads'

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
        image.save(file_path)
        image.close()

def process_img(filename):
    original = load_img(filename, target_size=(64, 64))
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)

    return image_batch

def model_predict(img_path, model):
    im = process_img(img_path)
    preds = model.predict(im)
    top_3 = preds.argsort()[0][::-1][:3]
    top_3_names = class_names[top_3]
    top_3_percent = preds[0][[top_3]]*100
    top_3_text = '<br>'.join([f'{name}: {percent:.2f}%' for name, percent in zip(top_3_names, top_3_percent)])
    return top_3_text


# home page
@app.route('/', methods=['GET', 'POST'])
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
    if request.method == 'GET':
        
        f = request.files['file']
        
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        rotate_save(f, file_path)
        preds = model_predict(file_path, model)

        os.remove(file_path)
        return render_template('predict.html', data=preds)
    return None



if __name__ == '__main__':

    model_path = "../models/model_acc2713.h5"

    model = load_model(model_path)
    model._make_predict_function()

    with open('../pickles/class_names.p', 'rb') as f:
                class_names = np.array(pickle.load(f))
    
    with open('../pickles/class_names_gen1_grouped.p', 'rb') as f:
                class_names = np.array(pickle.load(f))

    app.run(host='0.0.0.0', port=8080, threaded=True, debug=False)
