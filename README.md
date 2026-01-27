<div align="center">
<h1>Thermal3R: Physics-Induced Feed-forward Framework for Thermal 3D Reconstruction</h1>

</div>

## Overview

Thermal3R is the first feed-forward model designed for robust 3D reconstruction from thermal infrared images, capable of predicting high-precision depth maps, camera parameters, and point clouds from input infrared video sequences.


## Quick Start

First, clone this repository to your local machine, and install the dependencies (torch, torchvision, numpy, Pillow, and huggingface_hub). 

```bash
cd Thermal3R
pip install -r requirements.txt
```

If the VGGT code environment is already configured on your computer, it can be utilized directly.

## Dataset

The dataset will be shared via Google Drive and will be made publicly available in the near future. After you have download the dataset, you should place them in the root directory.

## Training the model

Our model supports both single-GPU and multi-GPU training.

The command for multi-GPU training is:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --use_env \
    training/launch.py \
    --your_training_arg1 value1 \
    --your_training_arg2 value2
```

## Testing the model

Upon completion of training, the obtained weights (named model.pt) can be placed in the pretrained folder. Alternatively, you may modify the corresponding path in demo_thermal.py and then proceed to run it.

```bash
python demo_thermal.py
```



