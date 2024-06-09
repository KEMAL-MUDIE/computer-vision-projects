import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from utils import *
from modules import *
from ddpm_conditional import *
# Sampaling part
n = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNet_conditional(num_classes=10).to(device)
ckpt = torch.load("class1_models/DDPM_conditional/class1_ema_ckpt.pt", map_location=device)
model.load_state_dict(ckpt)
diffusion = Diffusion(img_size=64, device=device)
y = torch.Tensor([6] * n).long().to(device)
x = diffusion.sample(model, n, y, cfg_scale=1)

# Create a samples folder if it doesn't exist
samples_folder = 'condclass1_samples'
os.makedirs(samples_folder, exist_ok=True)

for i in range(min(n, x.size(0))):
    sample_image = x[i].detach().cpu()

    # Permute dimensions for visualization
    sample_image = torch.clamp(sample_image.permute(1, 2, 0), 0, 1)

    # Visualize the sample
    plt.imshow(sample_image)
    plt.title(f'Sample {i+1}')
    plt.show()

    # Save the sample image
    save_path = os.path.join(samples_folder, f'sample_{i+1}.png')
    # Save the permuted image
    save_image(sample_image.permute(2, 0, 1), save_path)

print("Samples generated and saved in the 'samples' folder.")