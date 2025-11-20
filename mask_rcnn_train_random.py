import os
import sys
import copy
import torch
import random
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger
from detectron2.model_zoo import get_config_file, get_checkpoint_url


# =======================
# Setup
# =======================
setup_logger()
print("Training started...")

# Dataset paths
dataset_folder = "/home/fahadabul/mask_rcnn_skyhub/dataset_separated"
train_json = os.path.join(dataset_folder, "train/annotations.json")
val_json = os.path.join(dataset_folder, "test/annotations.json")
train_images = os.path.join(dataset_folder, "train/images")
val_images = os.path.join(dataset_folder, "test/images")
output_dir = os.path.join(dataset_folder, "output")
os.makedirs(output_dir, exist_ok=True)

# Redirect logs
sys.stdout = open(os.path.join(output_dir, "training_log.txt"), "w")
sys.stderr = sys.stdout

# Register datasets
register_coco_instances("my_dataset_train", {}, train_json, train_images)
register_coco_instances("my_dataset_val", {}, val_json, val_images)


# =======================
# Custom Augmentation Mapper
# =======================
class CustomColorAugmentationMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format="BGR")

        # Apply standard augmentations
        if self.is_train:
            aug_input = T.AugInput(image)
            transforms = self.augmentations(aug_input)
            image = aug_input.image
        else:
            transforms = []

        # Apply random color shift
        if random.random() < 0.1:
            image = image.astype(np.int16)
            for c in range(3):  # BGR
                shift = np.random.randint(-50, 50)
                image[:, :, c] = np.clip(image[:, :, c] + shift, 0, 255)
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


# =======================
# Custom Trainer with Early Stopping
# =======================
# Custom trainer class inheriting from Detectron2's DefaultTrainer
class EarlyStoppingTrainer(DefaultTrainer):
    def __init__(self, cfg, patience=50):
        super().__init__(cfg)  # Call the DefaultTrainer's initializer

        # Track the best metric (e.g., AP) seen so far
        self.best_metric = 0.0

        # Number of evaluation periods with no improvement
        self.no_improve_count = 0

        # Maximum allowed evaluations without improvement before early stopping
        self.patience = patience

        # Path to save the best model
        self.best_model_path = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")

    # Override method to build evaluator for validation (COCO-style in this case)
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")

        # Ensure the output directory exists
        os.makedirs(output_folder, exist_ok=True)

        # Return a COCO-style evaluator
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    # Override method to create training data loader with optional custom mapper
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=CustomColorAugmentationMapper(cfg, is_train=True))

    # Custom train method with early stopping logic
    def train(self):
        super().train()  # Initializes training state (like self.start_iter)

        max_iter = self.cfg.SOLVER.MAX_ITER  # Total training iterations
        eval_period = self.cfg.TEST.EVAL_PERIOD  # How often to evaluate (in iterations)

        # Main training loop
        for iteration in range(self.start_iter, max_iter):
            self.optimizer.zero_grad()       # Clear gradients
            loss_dict = self.run_step()      # Perform a forward + backward pass
            self.optimizer.step()            # Update model weights
            self.scheduler.step()            # Update learning rate scheduler

            # Run evaluation either at eval_period or at final iteration
            if (iteration + 1) % eval_period == 0 or (iteration + 1) == max_iter:
                print(f"\nEvaluating at iteration {iteration + 1}...")

                # Run evaluation using the test dataset and COCOEvaluator
                results = self.test(
                    self.cfg,
                    self.model,
                    evaluators=[self.build_evaluator(self.cfg, self.cfg.DATASETS.TEST[0])]
                )

                # Choose metric to monitor for early stopping
                curr_metric = results["segm"]["AP"]  # e.g., Average Precision for segmentation

                # If new best metric found, save model
                if curr_metric > self.best_metric:
                    print(f"New best AP: {curr_metric:.4f} (prev: {self.best_metric:.4f}) — Saving model.")
                    self.best_metric = curr_metric
                    self.no_improve_count = 0
                    torch.save(self.model.state_dict(), self.best_model_path)

                # No improvement in metric
                else:
                    self.no_improve_count += 1
                    print(f"No improvement. Patience count: {self.no_improve_count}/{self.patience}")

                # If early stopping condition met, break training loop
                if self.no_improve_count >= self.patience:
                    print(f"Early stopping triggered after {self.patience} rounds with no improvement.")
                    break

        print(f"Training complete. Best model saved at {self.best_model_path}")


# =======================
# Config Setup
# =======================
def get_cfg_with_settings():
    cfg = get_cfg()
    cfg.merge_from_file(get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.MODEL.WEIGHTS = get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 7700
    cfg.SOLVER.STEPS = (5100, 6800)
    cfg.SOLVER.GAMMA = 0.1

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only wrinkle class

    cfg.TEST.EVAL_PERIOD = 500
    cfg.OUTPUT_DIR = output_dir
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    return cfg


# =======================
# Main Execution
# =======================
if __name__ == "__main__":
    cfg = get_cfg_with_settings()
    trainer = EarlyStoppingTrainer(cfg, patience=5)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Save final model after training ends
    final_model_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    torch.save(trainer.model.state_dict(), final_model_path)
    print(f"Training complete. Final model saved at {final_model_path}")
