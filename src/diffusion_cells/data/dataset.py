
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms


# Create a DataLoader
class PBCCellDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir: path to e.g. 'train/' directory
        """
        self.root_dir = Path(root_dir)

        # Collect all image paths recursively
        self.image_paths = sorted( p for p in self.root_dir.rglob("*.jpg"))

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {root_dir}")

        self.transform = transforms.Compose([
            #transforms.Resize((64, image_size)),
            #transforms.RandomHorizontalFlip() # it's not really needed for HandE images, cells are roughly rotationally symmetric
            transforms.ToTensor(),                 # scales data to [0, 1]
            transforms.Lambda(lambda t: t * 2 - 1) # scales data to [-1, 1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        x0 = self.transform(img)
        return x0