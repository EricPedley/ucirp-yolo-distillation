from ultralytics import YOLO

task='detection'
model = YOLO('yolov8n.yaml')
    
model.train(
    data='yolo_output/metadata.yaml', 
    epochs=100, 
    save=True,
    workers=4,
    cos_lr=True,
)