from flask import Flask, render_template, request, jsonify, send_file
import os
import cv2
from io import BytesIO
from PIL import Image
import numpy as np

from sklearn.datasets import load_files
from app_model import object_detection
from app_model import yolo_preds_for_real_time

# webserver gateway interface
app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')
DEFAULT_UPLOAD_PATH = os.path.join(BASE_PATH,'static/Verification/')
PREDICTIONS_PATH = os.path.join(BASE_PATH,'static/predict/')


@app.route('/',methods=['POST','GET'])
def index():
    return render_template('index.html')

@app.route('/process_uploaded_image', methods=['POST'])
def process_uploaded_image():
    # get uploaded file
    image = request.files['image']
    filename = image.filename

    path_save = os.path.join(UPLOAD_PATH,filename)
    image.save(path_save)

    text = object_detection(path_save, filename)
    print(text)
    
    return jsonify({
        'filename': filename,
        'text': text
    })

@app.route('/process_image', methods=['POST'])
def process_image():
    filename = request.form['file_name']

    path_save = os.path.join(DEFAULT_UPLOAD_PATH,filename)

    text = object_detection(path_save, filename)
    print(text)

    # Return the result to the front-end
    return jsonify(text=text)


@app.route('/rtod', methods=['POST', 'GET'])
def rtod():
    return render_template('video.html')

@app.route('/detect', methods=['POST'])
def detect():
    # Read image data from request
    print('inside')
    file = request.files['image']
    filename = file.filename
    print(filename)

    path_save = os.path.join(UPLOAD_PATH,filename)
    path = os.path.join(path_save, '.jpeg')
    # image.save(path_save)
    
    # image = cv2.imread(path_save)
    # image = np.array(image,dtype=np.uint8) 

    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    print('inside1')
    # text = object_detection(path_save, filename)
    # print(text)

    boxes_np, confidences_np, index = yolo_preds_for_real_time(image)
    print('inside2')

    # Format detection results as JSON
    results = []
    for ind in index:
        left, top, w, h = boxes_np[ind]
        results.append({'label': 'NumberPlate', 'confidence': float(confidences_np[ind]), 'x': int(left), 'y': int(top), 'w': int(w), 'h': int(h)})
    
    # print(results)
    print('final')

    return jsonify(results)

if __name__ =="__main__":
    app.run(host ='0.0.0.0', debug=True)
    # host ='0.0.0.0', port = 8080,