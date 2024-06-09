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
        image = Image.open(img_path)#.convert("RGB")

        # Extract class information from the last 4 characters of the file name
        class_name = img_name[-5:-4]  # Assuming "220106_A207_4.jpg" format

        # Convert class_name to a numerical label if needed
        label = int(class_name)  # Assuming class_name is a single digit

        if self.transform:
            image = self.transform(image)

        return image, label  # or return image, label if using numerical labels
