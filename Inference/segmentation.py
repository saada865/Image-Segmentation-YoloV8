from ultralytics import YOLO


model = YOLO('../yolov8n-seg.pt')

results = model("img/testImg.jpg", save=True)

print(results)
