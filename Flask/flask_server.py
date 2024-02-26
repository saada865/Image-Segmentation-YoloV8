from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO('../yolov8n-seg.pt')


@app.route('/segment', methods=['POST'])
def segment_image():
    # os.remove('runs/segment/predict8')

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    input_file = request.files['image']
    # input_file = request.files.get('image')
    # os.system('mkdir test2')
    input_file.save('test2/temp.jpg')

    results = model(input_file.filename, save=True, project="runs", name="predict9")
    # results.save('test3/temp.jpg')
    # Return the segmentation results
    return send_file('../runs/predict9/testImg.jpg')


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
