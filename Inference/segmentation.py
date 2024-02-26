from ultralytics import YOLO


def segment(img_path):
    """
    This function performs image segmentation on the user image provided.

    param (img_path):
    This is the user provided image file path.
    """
    model = YOLO('../yolov8n-seg.pt')

    results = model(img_path, save=True)

    print(results)


segment("img/bigstock-Kids-Play-Football-Cute-Littl-471646067.jpg")
