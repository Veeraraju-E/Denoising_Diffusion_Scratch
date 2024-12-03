import torch
from PIL.Image import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class MNISTDataset(Dataset):
    def __init__(self, images_path):
        super().__init__()