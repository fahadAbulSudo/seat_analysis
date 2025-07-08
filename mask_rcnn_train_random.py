from detectron2.data import build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper
import numpy as np
import random
import torch
import copy
import cv2
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger
from detectron2.model_zoo import get_config_file, get_checkpoint_url
import sys

# Setup logger
setup_logger()
print("Training started...")

# Paths
dataset_folder = "/home/fahadabul/mask_rcnn_skyhub/dataset_separated"
train_json = os.path.join(dataset_folder, "train/annotations.json")
val_json = os.path.join(dataset_folder, "test/annotations.json")
train_images = os.path.join(dataset_folder, "train/images")
val_images = os.path.join(dataset_folder, "test/images")
output_dir = os.path.join(dataset_folder, "output")

# Register datasets
register_coco_instances("my_dataset_train", {}, train_json, train_images)
register_coco_instances("my_dataset_val", {}, val_json, val_images)

# Metadata
train_metadata = MetadataCatalog.get("my_dataset_train")
val_metadata = MetadataCatalog.get("my_dataset_val")

# Create configuration
cfg = get_cfg()
cfg.merge_from_file(get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.DATALOADER.NUM_WORKERS = 4

cfg.MODEL.WEIGHTS = get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00025

cfg.SOLVER.MAX_ITER = 7700  # 50 epochs
cfg.SOLVER.STEPS = (5100, 6800)  # learning rate decay
cfg.SOLVER.GAMMA = 0.1

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # wrinkle only

cfg.TEST.EVAL_PERIOD = 500  # evaluate every 500 iters
cfg.OUTPUT_DIR = output_dir
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Redirect stdout/stderr to a log file
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(cfg.OUTPUT_DIR, "training_log.txt")
sys.stdout = open(log_file, "w")
sys.stderr = open(log_file, "w")

# Custom Mapper
class CustomColorAugmentationMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format="BGR")

        # Standard augmentations
        if self.is_train:
            aug_input = T.AugInput(image)
            transforms = self.augmentations(aug_input)
            image = aug_input.image

        # Random color change
        if random.random() < 0.1:
            print("Randomly changing full image color in this iteration.")
            image = image.astype(np.int16)  # <-- Cast to int16 first
            for c in range(3):  # BGR channels
                random_shift = np.random.randint(-50, 50)
                image[:, :, c] = np.clip(image[:, :, c] + random_shift, 0, 255)
            image = image.astype(np.uint8) 

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if self.is_train and "annotations" in dataset_dict:
            annos = [
                utils.transform_instance_annotations(obj, transforms, image.shape[:2])
                for obj in dataset_dict.pop("annotations")
            ]
            instances = utils.annotations_to_instances(annos, image.shape[:2])
            dataset_dict["instances"] = instances

        return dataset_dict

# Custom Trainer
class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")
        os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=CustomColorAugmentationMapper(cfg, is_train=True))

# Train model
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Save final model
final_model_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
torch.save(trainer.model.state_dict(), final_model_path)
print(f"Training complete. Final model saved at {final_model_path}")
