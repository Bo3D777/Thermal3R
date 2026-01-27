import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16  # or torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
# model = VGGT.from_pretrained("/data2/ssh/Thermal_VGGT/vggt-main/weights/model.pt").to(device)

model = VGGT().to(device)
state_dict = torch.load("/data2/ssh/Thermal_VGGT/vggt-main/weights/model.pt")
model.load_state_dict(state_dict)


# Load and preprocess example images (replace with your own image paths)
image_names = ["/data2/ssh/Thermal_VGGT/vggt-main/0007.png"]  
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad() and torch.cuda.amp.autocast(dtype=dtype):
    # Predict attributes including cameras, depth maps, and point maps.
    predictions = model(images)
    print(predictions)
