from ultralytics import YOLO
import supervision as sv
import cv2 as cv

model = YOLO('runs/detect/train5/weights/80.pt')

img = cv.imread("rocket.png")

results = model(img)[0]
detections = sv.Detections.from_ultralytics(results)

bounding_box_annotator = sv.BoundingBoxAnnotator()

annotated_image = bounding_box_annotator.annotate(
    scene=img, detections=detections)

cv.imwrite("yolo_annotated_rocket.png", annotated_image)