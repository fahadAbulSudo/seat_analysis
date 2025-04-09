import os
import torch
import mlflow
import detectron2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.config import get_cfg, CfgNode
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger
from detectron2.model_zoo import get_config_file, get_checkpoint_url

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
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MLflow Configuration
cfg.MLFLOW = CfgNode()
cfg.MLFLOW.EXPERIMENT_NAME = "Mask_RCNN_Experiment"
cfg.MLFLOW.RUN_NAME = "Mask_RCNN_Run_01"
cfg.MLFLOW.RUN_DESCRIPTION = "Training with 3750 iterations, batch size 4."
cfg.MLFLOW.TRACKING_URI = "http://localhost:5000"

# Create output directory
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Custom MLflow Hook
class MLflowHook(HookBase):
    """ A custom hook class that logs artifacts, metrics, and parameters to MLflow. """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()

    def before_train(self):
        """ Initialize MLflow tracking before training starts. """
        mlflow.set_tracking_uri(self.cfg.MLFLOW.TRACKING_URI)
        mlflow.set_experiment(self.cfg.MLFLOW.EXPERIMENT_NAME)
        mlflow.start_run(run_name=self.cfg.MLFLOW.RUN_NAME)

        mlflow.set_tag("mlflow.note.content", self.cfg.MLFLOW.RUN_DESCRIPTION)

        # Log all configuration parameters
        for k, v in self.cfg.items():
            if isinstance(v, (int, float, str, bool)):  # Avoid logging complex objects
                mlflow.log_param(k, v)

    def after_step(self):
        """ Log training metrics at every step. """
        storage = self.trainer.storage
        iter_num = storage.iter
        latest_metrics = storage.latest()
        
        for k, v in latest_metrics.items():
            if isinstance(v, (int, float)):  # Only log numerical values
                mlflow.log_metric(k, v, step=iter_num)

    def after_train(self):
        """ Log artifacts and save final model after training completes. """
        model_path = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")

        # Save model and training artifacts
        if os.path.exists(model_path):
            mlflow.log_artifact(model_path)
        
        mlflow.log_artifacts(self.cfg.OUTPUT_DIR)  # Log all training outputs

        # Save the final configuration file
        with open(os.path.join(self.cfg.OUTPUT_DIR, "model-config.yaml"), "w") as f:
            f.write(self.cfg.dump())

        mlflow.end_run()

# Custom Trainer with MLflow Hook
class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")
        os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    def __init__(self, cfg):
        super().__init__(cfg)
        self.register_hooks([MLflowHook(cfg)])

# Initialize and Train Model
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Save Trained Model
trained_model_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
print(f"Training complete. Model saved at: {trained_model_path}")
