import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import List, Dict, Tuple, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Augmentation:
    """
    Class for data augmentation techniques.

    ...

    Attributes
    ----------
    transform_config : Dict
        Configuration for data transformations.

    Methods
    -------
    apply_transforms(self, data: Dict[str, Union[np.ndarray, pd.DataFrame]]) -> Dict[str, Union[np.ndarray, pd.DataFrame]]
        Apply augmentation transforms to the input data.

    """

    def __init__(self, transform_config: Dict):
        """
        Initialize the Augmentation class with the provided configuration.

        Parameters
        ----------
        transform_config : Dict
            Configuration for data transformations, including enable/disable flags and parameters.

        """
        self.transform_config = transform_config

    def apply_transforms(self, data: Dict[str, Union[np.ndarray, pd.DataFrame]]) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
        """
        Apply augmentation transforms to the input data.

        Parameters
        ----------
        data : Dict[str, Union[np.ndarray, pd.DataFrame]]
            Input data to be augmented. Can contain numpy arrays or pandas DataFrames.

        Returns
        -------
        Dict[str, Union[np.ndarray, pd.DataFrame]]
            Augmented data with the same structure as the input.

        Raises
        ------
        ValueError
            If a required transform is disabled in the configuration.
        TypeError
            If the input data type is not supported.

        """
        # Apply transformations based on configuration
        augmented_data = {}
        for key, value in data.items():
            if self.transform_config['random_rotation'] and isinstance(value, np.ndarray):
                augmented_data[key] = self._random_rotate(value)
            elif self.transform_config['random_flip'] and isinstance(value, np.ndarray):
                augmented_data[key] = self._random_flip(value)
            elif self.transform_config['random_noise'] and isinstance(value, np.ndarray):
                augmented_data[key] = self._add_noise(value)
            elif self.transform_config['random_crop'] and isinstance(value, np.ndarray):
                augmented_data[key] = self._random_crop(value)
            else:
                # No transformation required or unsupported data type
                augmented_data[key] = value

        return augmented_data

    def _random_rotate(self, image: np.ndarray) -> np.ndarray:
        """
        Randomly rotate an image by a specified angle.

        Parameters
        ----------
        image : np.ndarray
            Input image to be rotated.

        Returns
        -------
        np.ndarray
            Rotated image.

        """
        angle = random.randint(self.transform_config['min_rotation_angle'], self.transform_config['max_rotation_angle'])
        rotated_image = transforms.functional.rotate(image, angle)
        return rotated_image

    def _random_flip(self, image: np.ndarray) -> np.ndarray:
        """
        Randomly flip an image horizontally or vertically.

        Parameters
        ----------
        image : np.ndarray
            Input image to be flipped.

        Returns
        -------
        np.ndarray
            Flipped image.

        """
        flip_direction = random.randint(1, 4)  # 1: vertical, 2: horizontal, 3: both, 4: none
        flipped_image = transforms.functional.hflip(image) if flip_direction == 2 else image
        flipped_image = transforms.functional.vflip(flipped_image) if flip_direction in [1, 3] else flipped_image
        return flipped_image

    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Add random Gaussian noise to an image.

        Parameters
        ----------
        image : np.ndarray
            Input image to which noise will be added.

        Returns
        -------
        np.ndarray
            Noisy image.

        """
        mean = self.transform_config['noise_mean']
        std = self.transform_config['noise_std']
        noisy_image = image + np.random.normal(mean, std, image.shape)
        return noisy_image.clip(0, 255)

    def _random_crop(self, image: np.ndarray) -> np.ndarray:
        """
        Randomly crop an image to a specified size.

        Parameters
        ----------
        image : np.ndarray
            Input image to be cropped.

        Returns
        -------
        np.ndarray
            Cropped image.

        """
        crop_size = random.choice(self.transform_config['crop_sizes'])
        top = random.randint(0, image.shape[0] - crop_size)
        left = random.randint(0, image.shape[1] - crop_size)
        bottom = top + crop_size
        right = left + crop_size
        cropped_image = image[top:bottom, left:right]
        return cropped_image

class AugmentationConfig:
    """
    Configuration class for data augmentation.

    ...

    Attributes
    ----------
    enable_random_rotation : bool
        Flag to enable/disable random rotation transformation.
    min_rotation_angle : int
        Minimum rotation angle in degrees.
    max_rotation_angle : int
        Maximum rotation angle in degrees.
    enable_random_flip : bool
        Flag to enable/disable random flip transformation.
    enable_random_noise : bool
        Flag to enable/disable adding random noise transformation.
    noise_mean : float
        Mean of the Gaussian noise distribution.
    noise_std : float
        Standard deviation of the Gaussian noise distribution.
    enable_random_crop : bool
        Flag to enable/disable random crop transformation.
    crop_sizes : List[int]
        List of possible crop sizes.

    Methods
    -------
    load_from_dict(config_dict: Dict) -> None
        Load configuration from a dictionary.

    """

    def __init__(self):
        """
        Initialize the AugmentationConfig class with default values.
        These values can be overridden by calling load_from_dict().

        """
        self.enable_random_rotation = True
        self.min_rotation_angle = -10
        self.max_rotation_angle = 10
        self.enable_random_flip = True
        self.enable_random_noise = True
        self.noise_mean = 0.0
        self.noise_std = 10.0
        self.enable_random_crop = True
        self.crop_sizes = [64, 128, 256]

    def load_from_dict(self, config_dict: Dict) -> None:
        """
        Load configuration from a dictionary, updating the existing values.

        Parameters
        ----------
        config_dict : Dict
            Dictionary containing configuration values.

        """
        if 'enable_random_rotation' in config_dict:
            self.enable_random_rotation = config_dict['enable_random_rotation']
        if 'min_rotation_angle' in config_dict:
            self.min_rotation_angle = config_dict['min_rotation_angle']
        if 'max_rotation_angle' in config_dict:
            self.max_rotation_angle = config_dict['max_rotation_angle']
        if 'enable_random_flip' in config_dict:
            self.enable_random_flip = config_dict['enable_random_flip']
        if 'enable_random_noise' in config_dict:
            self.enable_random_noise = config_dict['enable_random_noise']
        if 'noise_mean' in config_dict:
            self.noise_mean = config_dict['noise_mean']
        if 'noise_std' in config_dict:
            self.noise_std = config_dict['noise_std']
        if 'enable_random_crop' in config_dict:
            self.enable_random_crop = config_dict['enable_random_crop']
        if 'crop_sizes' in config_dict:
            self.crop_sizes = config_dict['crop_sizes']

def validate_and_process_config(config: Dict) -> Dict:
    """
    Validate and process the provided configuration dictionary.

    Parameters
    ----------
    config : Dict
        Configuration dictionary.

    Returns
    -------
    Dict
        Processed configuration dictionary.

    Raises
    ------
    ValueError
        If a required configuration value is missing or invalid.

    """
    # Validate and process configuration
    processed_config = {}
    if 'enable_random_rotation' in config and isinstance(config['enable_random_rotation'], bool):
        processed_config['random_rotation'] = config['enable_random_rotation']
    else:
        raise ValueError("Invalid or missing value for 'enable_random_rotation' in configuration.")

    if 'min_rotation_angle' in config and isinstance(config['min_rotation_angle'], int):
        if config['min_rotation_angle'] < -360 or config['min_rotation_angle'] > 0:
            raise ValueError("Invalid value for 'min_rotation_angle'. Must be between -360 and 0 degrees.")
        processed_config['min_rotation_angle'] = config['min_rotation_angle']
    else:
        raise ValueError("Invalid or missing value for 'min_rotation_angle' in configuration.")

    if 'max_rotation_angle' in config and isinstance(config['max_rotation_angle'], int):
        if config['max_rotation_angle'] < 0 or config['max_rotation_angle'] > 360:
            raise ValueError("Invalid value for 'max_rotation_angle'. Must be between 0 and 360 degrees.")
        processed_config['max_rotation_angle'] = config['max_rotation_angle']
    else:
        raise ValueError("Invalid or missing value for 'max_rotation_angle' in configuration.")

    # Similar validation and processing for other configuration options...

    return processed_config

def main():
    """
    Main function to demonstrate the usage of the Augmentation class.

    """
    # Load configuration from a file or API
    config_dict = {
        'enable_random_rotation': True,
        'min_rotation_angle': -10,
        'max_rotation_angle': 10,
        'enable_random_flip': True,
        # ... other configuration options ...
    }

    # Validate and process configuration
    processed_config = validate_and_process_config(config_dict)

    # Create AugmentationConfig object and load configuration
    augmentation_config = AugmentationConfig()
    augmentation_config.load_from_dict(processed_config)

    # Create Augmentation object
    augmentation = Augmentation(augmentation_config)

    # Example data
    data = {
        'image1': np.random.rand(256, 256, 3) * 255,
        'image2': np.random.rand(128, 128, 1) * 255,
        # ... other data ...
    }

    # Apply augmentation transforms
    augmented_data = augmentation.apply_transforms(data)

    # Use augmented data for training, visualization, etc.
    # ...

if __name__ == '__main__':
    main()