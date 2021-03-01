"""
Definition of ImageFolderDataset dataset class
and image-specific transform classes
"""

# pylint: disable=too-few-public-methods

import os

import numpy as np
from PIL import Image

from .base_dataset import Dataset


class ImageFolderDataset(Dataset):
    """CIFAR-10 dataset class"""
    def __init__(self, *args, transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.classes, self.class_to_idx = self._find_classes(self.root_path)
        self.images, self.labels = self.make_dataset(
            directory=self.root_path,
            class_to_idx=self.class_to_idx
        )
        self.transform = transform

    @staticmethod
    def _find_classes(directory):
        """
        Finds the class folders in a dataset
        :param directory: root directory of the dataset
        :returns: (classes, class_to_idx), where
          - classes is the list of all classes found
          - class_to_idx is a dict that maps class to label
        """
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    @staticmethod
    def make_dataset(directory, class_to_idx):
        """
        Create the image dataset by preparaing a list of samples
        :param directory: root directory of the dataset
        :param class_to_idx: A dict that maps classes to labels
        :returns: (images, labels) where:
            - images is a list containing paths to all images in the dataset
            - labels is a list containing one label per image
        """
        images, labels = [], []
        ########################################################################
        # TODO:                                                                #
        # Construct a list of all images in the dataset                        #
        # and a list of corresponding labels.                                  #
        # Hints:                                                               #
        #   - have a look at the "CIFAR-10: Image Dataset" notebook first      #
        #   - class_idx contains all classes and corresponding labels          #
        #   - images should only contain file paths, NOT the actual images     #
        #   - sort images by class and file name (ascending)                   #
        ########################################################################

        # pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        #cls, label = self._find_classes(directory)
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        num_classes = len(classes)
        
        for label, cls in enumerate(classes):
            samples_per_class = len(next(os.walk(os.path.join(directory,cls)))[2])
            for i in range(samples_per_class):
                image_path = os.path.join(directory,
                                          cls,
                                          str(i+1).zfill(4) + ".png") # e.g. cifar10/plane/0001.png
                images.append(image_path)
                labels.append(label)
        
        assert len(images) == len(labels)
        return images, labels

    def __len__(self):
        length = None
        ########################################################################
        # TODO:                                                                #
        # Return the length of the dataset (number of images)                  #
        ########################################################################

        # pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        
        length = len(self.images)
        
        return length

    @staticmethod
    def load_image_as_numpy(image_path):
        """Load image from image_path as numpy array"""
        return np.asarray(Image.open(image_path), dtype=float)

    def __getitem__(self, index):
        data_dict = None
        ########################################################################
        # TODO:                                                                #
        # create a dict of the data at the given index in your dataset         #
        # The dict should be of the following format:                          #
        # {"image": <i-th image>, "label": <label of i-th image>}              #
        # Hints:                                                               #
        #   - use load_image_as_numpy() to load an image from a file path      #
        #   - Make sure to apply self.transform to the image                   #
        ########################################################################

        # pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        
        label = self.labels[index]
        path = self.images[index]
        
        image = self.load_image_as_numpy(path)
        
        if self.transform is not None:
            image = self.transform(image)
        
        data_dict = {"image" : image, 
                     "label" : label}
        
        return data_dict


def compute_image_mean_and_std(images):
    """
    Calculate the per-channel image mean and standard deviation of given images
    :param images: numpy array of shape NxHxWxC
        (for N images with C channels of spatial size HxW)
    :returns: per-channels mean and std; numpy array of shape C
    """
    mean, std = None, None
    ########################################################################
    # TODO:                                                                #
    # Calculate the per-channel mean and standard deviation of the images  #
    # Hint: You can use numpy to calculate mean and standard deviation     #
    ########################################################################

    #pass

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    mean = np.mean(images, axis=(0,1,2))
    std = np.std(images, axis=(0, 1, 2))

    return mean, std


class RescaleTransform:
    """Transform class to rescale images to a given range"""
    def __init__(self, range_=(0, 1), old_range=(0, 255)):
        """
        :param range_: Value range to which images should be rescaled
        :param old_range: Old value range of the images
            e.g. (0, 255) for images with raw pixel values
        """
        self.min = range_[0]
        self.max = range_[1]
        self._data_min = old_range[0]
        self._data_max = old_range[1]

    def __call__(self, images):
        ########################################################################
        # TODO:                                                                #
        # Rescale the given images:                                            #
        #   - from (self._data_min, self._data_max)                            #
        #   - to (self.min, self.max)                                          #
        ########################################################################

        # pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        images = images / (self._data_max/self.max)

        return images


class NormalizeTransform:
    """
    Transform class to normalize images using mean and std
    Functionality depends on the mean and std provided in __init__():
        - if mean and std are single values, normalize the entire image
        - if mean and std are numpy arrays of size C for C image channels,
            then normalize each image channel separately
    """
    def __init__(self, mean, std):
        """
        :param mean: mean of images to be normalized
            can be a single value, or a numpy array of size C
        :param std: standard deviation of images to be normalized
             can be a single value or a numpy array of size C
        """
        self.mean = mean
        self.std = std

    def __call__(self, images):
        images = (images - self.mean) / self.std
        return images


class ComposeTransform:
    """Transform class that combines multiple other transforms into one"""
    def __init__(self, transforms):
        """
        :param transforms: transforms to be combined
        """
        self.transforms = transforms

    def __call__(self, images):
        for transform in self.transforms:
            images = transform(images)
        return images
