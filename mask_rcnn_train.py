import os
import torch
import detectron2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger
from detectron2.model_zoo import get_config_file, get_checkpoint_url
import sys



print("Training started...")
# Setup logger
setup_logger()

# Paths
dataset_folder = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn/dataset"
train_json = os.path.join(dataset_folder, "train_annotations.json")
val_json = os.path.join(dataset_folder, "val_annotations.json")
train_images = os.path.join(dataset_folder, "train")
val_images = os.path.join(dataset_folder, "val")
output_dir = os.path.join(dataset_folder, "output")

# Register datasets in Detectron2
register_coco_instances("my_dataset_train", {}, train_json, train_images)
register_coco_instances("my_dataset_val", {}, val_json, val_images)

# Get metadata
train_metadata = MetadataCatalog.get("my_dataset_train")
val_metadata = MetadataCatalog.get("my_dataset_val")

# Create configuration
cfg = get_cfg()
cfg.merge_from_file(get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))  # Load from Model Zoo
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Pretrained weights
cfg.SOLVER.IMS_PER_BATCH = 4  # Reduce if OOM occurs
cfg.SOLVER.BASE_LR = 0.00025  # Learning rate
cfg.SOLVER.MAX_ITER = 3750  # Number of iterations
cfg.SOLVER.STEPS = (2500, 3200)  # Learning rate decay steps
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # RoI batch size
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Change if multiple classes exist
cfg.TEST.EVAL_PERIOD = 500  # Evaluate every 500 iterations
cfg.OUTPUT_DIR = output_dir
cfg.MODEL.DEVICE = "cpu"
log_file = os.path.join(cfg.OUTPUT_DIR, "training_log.txt")
sys.stdout = open(log_file, "w")  # Redirect stdout to file
sys.stderr = open(log_file, "w")  # Redirect stderr to file (errors)
# Create output directory
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Trainer class
class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")
        os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

# Train model
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Save trained model
trained_model_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
print(f"Training complete. Model saved at: {trained_model_path}")
