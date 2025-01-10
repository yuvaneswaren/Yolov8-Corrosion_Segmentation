import wandb

# Log in with your Wandb API key
wandb.login(key='bb056693227f44fe316365c723b3ecb82a97439e')
from ultralytics import YOLO

# loading a pre-trained model
# if the first time loading a model, it will first download the model in the directory
# available pre-trained models are YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x
model = YOLO("yolov8s-seg.pt")

# will throw an exception if false
model._check_is_pytorch_model()

data_yaml_path = "data.yaml"

# Use 'cpu' for device since you don't have CUDA available
model.train(data=data_yaml_path,
            epochs=300,
            imgsz=50,
            device='cpu')


