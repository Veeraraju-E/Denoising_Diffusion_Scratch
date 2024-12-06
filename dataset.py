from PIL.Image import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import glob as glob

class MNISTDataset(Dataset):
    def __init__(self, images_path, split):
        super().__init__()
        self.split = split
        self.images_path = images_path
        self.images, self.labels = self.load_images(self.images_path)

    def __len__(self):
        return len(self.images)

    def load_images(self, image_path):
        images, labels = [], []
        loop = tqdm(os.listdir(image_path))

        for d_name in loop:
            for f_name in glob.glob(os.path.join(d_name, f_name, '*.png')):
                images.append(f_name)
                labels.append(int(d_name))
        
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        t1 = transforms.ToTensor()
        image = t1(image)

        image = 2*image - 1 # scale image to be between -1 and 1
        label = self.labels[index]

        return image, label