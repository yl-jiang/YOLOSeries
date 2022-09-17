import warnings
import abc

import numpy as np


class Generator(abc.ABC):

    @abc.abstractmethod
    def size(self):
        """
        Size of dataset.
        :return: int
        """
        # raise NotImplementedError('size method is not implemented.')
    @property
    @abc.abstractmethod
    def num_class(self):
        """
        Number of calss in the dataset
        :return: int
        """
        # raise NotImplementedError('num_class method is not implemented.')

    @abc.abstractmethod
    def has_label(self, label):
        """
        Return True if label is known label
        :param label: string
        :return: bool
        """
        # raise NotImplementedError('has_label method is not implemented.')

    @abc.abstractmethod
    def has_name(self, name):
        """
        Return True is name is a known class
        :param name: string
        :return: bool
        """
        # raise NotImplementedError('has_name method is not Implemented.')

    @abc.abstractmethod
    def name_to_label(self, name):
        """
        Map name to label.
        :param name: string
        :return: int
        """
        # raise NotImplementedError('name_to_label method is not implemented.')

    @abc.abstractmethod
    def label_to_name(self, label):
        """
        Map label to name.
        :param label: int
        :return: string
        """
        # raise NotImplementedError('label_to_name method is not implemented.')

    @abc.abstractmethod
    def img_aspect_ratio(self, img_index):
        """
        Compute the aspect ratio for an image with img_index.
        :param img_index: int
        :return: int
        """
        # raise NotImplementedError('img_aspect_ratio method is not implemented.')

    @abc.abstractmethod
    def img_path(self, img_index):
        """
        Get image path with image index.
        :param img_index:
        :return:
        """
        # raise NotImplementedError('img_path method is not implemented.')

    @abc.abstractmethod
    def load_img(self, img_index):
        """
        Load an image at the image index.
        :param img_index: int
        :return: ndarray
        """
        # raise NotImplementedError('load_img method is not implemented.')

    @abc.abstractmethod
    def load_annotations(self, img_index):
        """
        Load annotations for an image index
        :param img_index: int
        :return: dict
        """
        # raise NotImplementedError('load_annotations method is not implemented.')

    def check_annotations(self, img, annotations, img_index):
        """
        Check annotations that are outside of image boundary or whose width/height < 0
        :param img: ndarray
        :param annotations: dict
        :return: dict
        """
        invalid_indices = np.where(
            (annotations['bboxes'][:, 2] <= annotations[:, 0]) |
            (annotations['bboxes'][:, 1] <= annotations[:, 3]) |
            (annotations['bboxes'][:, 2] > img.shape[1]) |
            (annotations['bboxes'][:, 3] > img.shape[0]) |
            (annotations['bboxes'][:, 0] < 0) |
            (annotations['bboxes'][:, 1] < 0)
        )[0]

        # delete invalid indices
        if len(invalid_indices):
            warnings.warn(f'Image {self.img_path(img_index)} with id {img_index} (shape {img.shape}) '
                          f'contains the following invalid boxes: {annotations[invalid_indices, :]}')
            np.delete(annotations, invalid_indices, axis=0)
        return img, annotations



