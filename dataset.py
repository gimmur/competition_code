import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import glob, os
import numpy as np

class Query_Images(torch.utils.data.Dataset):

    """create a Dataset class which returns the same image,
    len of instances and returns the same image in getitem.
    NO AUGMENTATION BECAUSE THIS GOES FOR TESTING"""

    def __init__(self, root: str, transform = None):
        self.root = root
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(self.root, "*.jpg")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image_name = os.path.basename(image_path)
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, image_name


def get_query_dataset(config):

    """define transformations and instantiate
    the dataset class  with transformations."""

    # Define the transformations at test time
    #transform_rgb = transforms.Lambda(lambda image: image.convert('RGB')) SHOULD NOT BE NECESSARY, RIGHT?
    transform = transforms.Compose([
        #transform_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0], std=[1]),
        transforms.Resize((224, 224))
    ])

    query = Query_Images(root = config["dataset_query"]["data_root_query"], transform = transform)

    # instantiate DataLoader:
    query_dataloader = torch.utils.data.DataLoader(query, config["dataset_query"]["batch_size_query"], shuffle = False)

    return query_dataloader

class Gallery_Images(torch.utils.data.Dataset): #see lab02 for transformations

    """create a Dataset class which returns the same image,
    len of instances, and returns the same image in getitem.
    NO AUGMENTATION BECAUSE THIS GOES FOR TESTING"""

    def __init__(self, root: str, transform = None):
        self.root = root
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(self.root, "*.jpg")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image_name = os.path.basename(image_path)
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, image_name


def get_gallery_dataset(config):

    """define transformations and instantiate
    the dataset class  with transformations."""

    # Define the transformations at test time
    #transform_rgb = transforms.Lambda(lambda image: image.convert('RGB')) SHOULD NOT BE NECESSARY, RIGHT?
    transform = transforms.Compose([
        #transform_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0], std=[1]),
        transforms.Resize((224, 224))
    ])

    gallery = Gallery_Images(root = config["dataset_gallery"]["data_root_gallery"], transform = transform)

    # instantiate DataLoader:
    gallery_dataloader = torch.utils.data.DataLoader(gallery, config["dataset_gallery"]["batch_size_gallery"], shuffle = False)

    return gallery_dataloader