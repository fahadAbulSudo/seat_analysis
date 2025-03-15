import os
import torch
import detectron2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger
from detectron2.model_zoo import get_config_file, get_checkpoint_url
import numpy as np
import time
# from google.colab import drive
import json

# %%
drive_folder = "/home/fahadabul/mask_rcnn_skyhub/final_images" 
os.makedirs(drive_folder, exist_ok=True)

print("Training started...")
setup_logger()

# %%
# Paths
dataset_folder = "/home/fahadabul/mask_rcnn_skyhub/final_images"  # Update this if dataset is elsewhere
train_json ='/home/fahadabul/mask_rcnn_skyhub/final_images/train_annotations_fixed.json' # os.path.join(dataset_folder, "train_annotations.json")
val_json ='/home/fahadabul/mask_rcnn_skyhub/final_images/val_annotations_fixed.json'  #os.path.join(dataset_folder, "val_annotations.json")
train_images = os.path.join(dataset_folder, "train")
val_images = os.path.join(dataset_folder, "val")
output_dir = os.path.join(drive_folder, "output")  # Save output to Google Drive
# %%
# Register datasets
def verify_coco_segmentation_format(json_path):
    """
    Verifies that the segmentation format in COCO json is in the expected format
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    for ann in data['annotations']:
        if 'segmentation' in ann:
            segm = ann['segmentation']
            # print("$$$$$$$$$$$$$$$$$$$$",type(segm))
            if not isinstance(segm, list):
                raise ValueError(f"Segmentation data is not a list for annotation ID {ann['id']}. The segmentation should be a list of polygons.")
            if not segm:
                continue
            if isinstance(segm[0], list):# and isinstance(segm[0][0], float):
                 continue
            elif isinstance(segm[0], dict) and 'counts' in segm[0]:
              continue
            else:
                raise ValueError(f"Segmentation format is not valid for annotation ID {ann['id']}. Should be a list of polygon points or a list of RLE objects")

print("Verifying training annotation file...")
verify_coco_segmentation_format(train_json)
print("Training annotation file verified!")

print("Verifying validation annotation file...")
verify_coco_segmentation_format(val_json)
print("Validation annotation file verified!")
register_coco_instances("my_dataset_train", {}, train_json, train_images)
register_coco_instances("my_dataset_val", {}, val_json, val_images)
# %%
# Get metadata
train_metadata = MetadataCatalog.get("my_dataset_train")
val_metadata = MetadataCatalog.get("my_dataset_val")

# %%
# Configuration
cfg = get_cfg()
cfg.merge_from_file(get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.DATALOADER.NUM_WORKERS = 2  # Reduce for Colab
cfg.MODEL.WEIGHTS = get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2  # Reduce for Colab
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 3750
cfg.SOLVER.STEPS = (2500, 3200)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.TEST.EVAL_PERIOD = 500
cfg.OUTPUT_DIR = output_dir
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# %%
# Create output directory
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# %%
# Early Stopping & Model Saving Hook
class EarlyStoppingHook(HookBase):
    def __init__(self, patience=3):  # Stop if no improvement for 3 evals
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0

    def after_step(self):
        if self.trainer.iter % cfg.TEST.EVAL_PERIOD == 0:
            eval_results = self.trainer.storage.latest()
            val_loss = eval_results.get("total_loss", None)

            print("val_loss", val_loss)

            # Check if val_loss is not None and is a tuple, then access the first element
            if val_loss is not None:
                val_loss_value = val_loss[0]  # Access the actual validation loss value

                # Print the validation loss correctly
                print(f"Iteration {self.trainer.iter}: Validation Loss = {val_loss_value:.4f}")

                model_path = os.path.join(cfg.OUTPUT_DIR, f"model_iter_{self.trainer.iter}.pth")
                torch.save(self.trainer.model.state_dict(), model_path)
                print(f"Model saved at: {model_path}")

                # Compare and save the best model based on the loss
                if val_loss_value < self.best_loss:
                    self.best_loss = val_loss_value
                    self.counter = 0
                    # Save the best model
                    torch.save(self.trainer.model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "best_model.pth"))
                else:
                    self.counter += 1
                    if self.counter >= self.patience:
                        print("Early stopping triggered. Stopping training.")
                        self.trainer.iter = cfg.SOLVER.MAX_ITER  # Force training to stop
# %%
# Custom Trainer
class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")
        os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.append(EarlyStoppingHook(patience=3))  # Add early stopping & model saving
        return hooks
# %%
# Train
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
# %%
# Save final model
final_model_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
torch.save(trainer.model.state_dict(), final_model_path)
print(f"Training complete. Model saved at: {final_model_path}")
