import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from utils import *
from modules import *
from ddpm import *
# Assuming you have the Diffusion and UNet classes and other necessary functions defined

# Load pre-trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
ckpt = torch.load("class_models/DDPM_Uncondtional/class1n_ckpt.pt", map_location=device)
model.load_state_dict(ckpt)
model.to(device)
diffusion = Diffusion(img_size=64, device=device)

# Create a samples folder if it doesn't exist
samples_folder = 'class1(161+)_samples'
os.makedirs(samples_folder, exist_ok=True)

# Generate 20 samples
num_samples = 3
generated_samples = diffusion.sample(model, n=num_samples)
#plot_images(generated_samples)
# Visualize and save the samples in a 5x4 subplot
num_columns = 3
num_rows = 1

# Create a figure and axis
fig, axs = plt.subplots(num_rows, num_columns, figsize=(8, 6))
# Iterate over samples
for i in range(num_samples):
    sample_image = generated_samples[i].detach().cpu()
    # Convert sample_image to torch.float and normalize to [0, 1]
    sample_image = sample_image.float() / 255.0  # Assuming the values are in the range [0, 255]
    # Compute subplot indices
    row_idx = i // num_columns
    col_idx = i % num_columns

    # Plot the sample in the subplot
    axs[row_idx, col_idx].imshow(torch.clamp(sample_image.permute(1, 2, 0), 0, 1))
    axs[row_idx, col_idx].set_title(f'Sample {i+201}')
    axs[row_idx, col_idx].axis('off')
    # Save the sample image
    save_path = os.path.join(samples_folder, f'{i+201}220105_A095_2.png')
    save_image(sample_image, save_path)

# Adjust layout to prevent clipping of titles
plt.tight_layout()

# Show the generated samples subplot
plt.show()

print("Samples generated and saved in the 'samples' folder.")