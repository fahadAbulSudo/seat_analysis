# Mask R-CNN Installation using Detectron2

This guide provides step-by-step instructions for installing Mask R-CNN using Detectron2 from GitHub.

## Prerequisites
Ensure you have the following installed:
- Python 3.8+
- PyTorch (compatible with your CUDA version)
- NVIDIA GPU (optional but recommended for training)
- CUDA and cuDNN (if using GPU acceleration)
- Git

## Step 1: Clone Detectron2 Repository
```bash
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
```

## Step 2: Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv detectron_env
source detectron_env/bin/activate  # On Windows use: detectron_env\Scripts\activate
```

## Step 3: Install Dependencies
```bash
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

## Step 4: Install PyTorch
Ensure you install the correct PyTorch version based on your CUDA version. Check [PyTorch official site](https://pytorch.org/get-started/locally/) for the latest versions.

For example, if using CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CPU-only installation:
```bash
pip install torch torchvision torchaudio
```

## Step 5: Install Detectron2
### Installing from GitHub (Recommended for latest updates)
```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Or, for a specific version (replace `<SHA>` with a commit hash or branch name):
```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git@<SHA>'
```

## Step 6: Verify Installation
```bash
python -c "import detectron2; print(detectron2.__version__)"
```
If there are no errors, the installation is successful.

## Step 7: Run a Sample Test
Test Mask R-CNN with a sample image:
```bash
python demo/demo.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --input demo/sample.jpg --output output.jpg --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
```

## Additional Resources
- [Detectron2 Official Documentation](https://detectron2.readthedocs.io/)
- [Detectron2 GitHub Repository](https://github.com/facebookresearch/detectron2)

## Troubleshooting
If you encounter issues, refer to the official documentation or check common problems on GitHub Discussions.

---
This setup ensures that you have Mask R-CNN installed using Detectron2 with minimal hassle. Enjoy experimenting with instance segmentation!

