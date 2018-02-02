from settings import (
    RAW_IMAGE_DIRECTORY, RAW_IMAGE_NAMING
)

import Augmentor
import numpy as np
import os
import random
import glob


class CharacterAugmentor(object):

    def __init__(self, input_root=RAW_IMAGE_DIRECTORY):
        self.input_root = os.path.abspath(input_root)
        # Directories organized by character
        self.directories = glob.glob(os.path.join(self.input_root, '*'))

        self.hex_class_labels = [self._get_label_from_dir(x) for x in self.directories]
        # Define self.hex_class_label_mapping
        self._construct_hex_mapping()
        # Define self.pipelines
        self._construct_pipelines()

    def _get_label_from_dir(self, directory):
        return os.path.split(directory)[1]

    def _construct_hex_mapping(self):
        self.hex_class_label_mapping = {
            hex_label: hex_label.decode('hex')
            for hex_label in self.hex_class_labels
        }

    def _add_augmentation_layers(self, pipeline):
        pipeline.rotate_random_90(probability=0.5)
        pipeline.shear(probability=0.5, max_shear_left=10, max_shear_right=10)
        pipeline.random_distortion(
            probability=0.8, grid_width=3, grid_height=3, magnitude=3
        )

    def _construct_pipelines(self):
        self.pipelines = {}
        for directory in self.directories:
            label = self._get_label_from_dir(directory)
            self.pipelines[label] = Augmentor.Pipeline(directory)
            self._add_augmentation_layers(self.pipelines[label])
