import os
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.file_list = os.listdir(root)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.root, img_name)
        image = Image.open(img_path).convert("RGB")

        # Extract class information from the file name
        class_name = img_name.split("_")[0]  # Assuming "class_xxx.jpg" format

        # Convert class_name to a numerical label if needed
        # label = ...

        if self.transform:
            image = self.transform(image)

        return image, class_name  # or return image, label if using numerical labels
