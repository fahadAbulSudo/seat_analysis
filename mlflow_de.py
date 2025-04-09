import mlflow
import torch
import os
import yaml
from detectron2.config import get_cfg
from detectron2.model_zoo import get_config_file
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model

# Paths
MODEL_PATH = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn/dataset/output/model_final_torn_second.pth"
MLFLOW_EXPERIMENT_NAME = "Mask_RCNN_Deployment"
ARTIFACT_PATH = "detectron2_mask_rcnn"

# Define class names
CLASS_NAMES = ["torn"]

# Setup MLflow
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# Configure Detectron2 Model
cfg = get_cfg()
cfg.merge_from_file(get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)
cfg.MODEL.WEIGHTS = MODEL_PATH
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load Model as a PyTorch Module
model = build_model(cfg)  # This is a valid `torch.nn.Module`
model.load_state_dict(torch.load(MODEL_PATH, map_location=cfg.MODEL.DEVICE))
model.eval()

# Save config file separately
config_path = "detectron2_config.yaml"
with open(config_path, "w") as f:
    f.write(cfg.dump())

# Start MLflow Run
with mlflow.start_run():
    # Log Model
    mlflow.pytorch.log_model(model, ARTIFACT_PATH)

    # Log Config File
    mlflow.log_artifact(config_path)

    # Log Parameters
    mlflow.log_params({
        "framework": "Detectron2",
        "num_classes": len(CLASS_NAMES),
        "score_threshold": cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
        "device": cfg.MODEL.DEVICE
    })

print("✅ Mask R-CNN model successfully logged in MLflow!")
