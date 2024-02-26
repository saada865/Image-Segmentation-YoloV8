import os
import shutil

from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO('yolov8n-seg.pt')


@app.route('/segment', methods=['POST'])
def segment_image():
    if os.path.exists("runs/predict9/temp.jpg"):
        os.remove('runs/predict9/temp.jpg')
        shutil.rmtree('runs/predict9')

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    input_file = request.files['image']
    input_file.save('test2/temp.jpg')

    results = model('test2/temp.jpg', save=True, project="runs", name="predict9")

    return send_file('runs/predict9/temp.jpg')


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
