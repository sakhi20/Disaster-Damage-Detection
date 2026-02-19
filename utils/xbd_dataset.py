import os
import cv2
import torch
from torch.utils.data import Dataset

class XBDDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.pre_imgs = {}
        self.post_imgs = {}

        for f in os.listdir(image_dir):
            if "pre_disaster" in f:
                key = f.replace("_pre_disaster.png", "")
                self.pre_imgs[key] = f
            elif "post_disaster" in f:
                key = f.replace("_post_disaster.png", "")
                self.post_imgs[key] = f

        self.keys = sorted(list(set(self.pre_imgs) & set(self.post_imgs)))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]

        pre = cv2.imread(os.path.join(self.image_dir, self.pre_imgs[key]))
        post = cv2.imread(os.path.join(self.image_dir, self.post_imgs[key]))

        pre = cv2.cvtColor(pre, cv2.COLOR_BGR2RGB)
        post = cv2.cvtColor(post, cv2.COLOR_BGR2RGB)

        pre = torch.tensor(pre).permute(2, 0, 1).float() / 255.0
        post = torch.tensor(post).permute(2, 0, 1).float() / 255.0

        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, key + "_post_disaster.png")
            mask = cv2.imread(mask_path, 0)
            # Handle cases where mask might be missing or different
            if mask is None:
                # Return zero mask if missing (or handle appropriately)
                mask = torch.zeros((pre.shape[1], pre.shape[2])).long()
            else:
                mask = torch.tensor(mask).long()
                # For classification in notebook, mask > 0 was used sometimes. 
                # Here we keep raw indices for CrossEntropy.
            return pre, post, mask

        return pre, post
