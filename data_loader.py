import logging
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import List, Tuple, Dict

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoaderConfig:
    """
    Configuration class for data loader.
    
    Attributes:
    batch_size (int): Batch size for data loading.
    num_workers (int): Number of worker threads for data loading.
    pin_memory (bool): Whether to pin memory for data loading.
    """
    def __init__(self, batch_size: int = 32, num_workers: int = 4, pin_memory: bool = True):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

class ImageDataset(Dataset):
    """
    Custom dataset class for image data.
    
    Attributes:
    image_paths (List[str]): List of image file paths.
    labels (List[int]): List of corresponding labels.
    transform (transforms.Compose): Data transformation pipeline.
    """
    def __init__(self, image_paths: List[str], labels: List[int], transform: transforms.Compose):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        
        Returns:
        int: Number of samples.
        """
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Returns a sample from the dataset.
        
        Args:
        index (int): Index of the sample.
        
        Returns:
        Tuple[torch.Tensor, int]: Sample image and label.
        """
        image_path = self.image_paths[index]
        label = self.labels[index]
        
        try:
            image = np.load(image_path)
            image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None, None

class DataLoaderException(Exception):
    """
    Custom exception class for data loader errors.
    """
    pass

class DataLoader:
    """
    Data loader class for image data.
    
    Attributes:
    config (DataLoaderConfig): Configuration for data loader.
    dataset (ImageDataset): Custom dataset instance.
    data_loader (DataLoader): PyTorch data loader instance.
    """
    def __init__(self, config: DataLoaderConfig, dataset: ImageDataset):
        self.config = config
        self.dataset = dataset
        self.data_loader = self._create_data_loader()

    def _create_data_loader(self) -> DataLoader:
        """
        Creates a PyTorch data loader instance.
        
        Returns:
        DataLoader: PyTorch data loader instance.
        """
        try:
            data_loader = DataLoader(
                self.dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            )
            return data_loader
        except Exception as e:
            logger.error(f"Error creating data loader: {str(e)}")
            raise DataLoaderException("Error creating data loader")

    def load_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads a batch of data from the data loader.
        
        Returns:
        Tuple[torch.Tensor, torch.Tensor]: Batch of images and labels.
        """
        try:
            batch = next(iter(self.data_loader))
            images, labels = batch
            return images, labels
        except Exception as e:
            logger.error(f"Error loading batch: {str(e)}")
            raise DataLoaderException("Error loading batch")

def create_image_dataset(image_paths: List[str], labels: List[int], transform: transforms.Compose) -> ImageDataset:
    """
    Creates a custom image dataset instance.
    
    Args:
    image_paths (List[str]): List of image file paths.
    labels (List[int]): List of corresponding labels.
    transform (transforms.Compose): Data transformation pipeline.
    
    Returns:
    ImageDataset: Custom image dataset instance.
    """
    return ImageDataset(image_paths, labels, transform)

def create_data_loader(config: DataLoaderConfig, dataset: ImageDataset) -> DataLoader:
    """
    Creates a data loader instance.
    
    Args:
    config (DataLoaderConfig): Configuration for data loader.
    dataset (ImageDataset): Custom dataset instance.
    
    Returns:
    DataLoader: Data loader instance.
    """
    return DataLoader(config, dataset)

def main():
    # Example usage
    image_paths = ["image1.npy", "image2.npy", "image3.npy"]
    labels = [0, 1, 0]
    transform = transforms.Compose([transforms.ToTensor()])
    
    dataset = create_image_dataset(image_paths, labels, transform)
    config = DataLoaderConfig(batch_size=32, num_workers=4, pin_memory=True)
    data_loader = create_data_loader(config, dataset)
    
    images, labels = data_loader.load_batch()
    logger.info(f"Loaded batch of {images.shape[0]} images and {labels.shape[0]} labels")

if __name__ == "__main__":
    main()