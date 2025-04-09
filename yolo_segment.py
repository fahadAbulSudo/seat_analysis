from ultralytics import YOLO

# Load a pretrained YOLOv8 segmentation model
model = YOLO('yolov8n-seg.pt')

# Train the model
model.train(data='/home/fahadabul/mask_rcnn_skyhub/segment/data.yaml',
            epochs=50,
            imgsz=640,
            batch=16)

metrics = model.val()
print(metrics)