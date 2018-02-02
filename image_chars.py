from good_font_list import goodFonts
from settings import (
    RAW_IMAGE_DIRECTORY, RAW_IMAGE_NAMING
)

from glob import glob
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.interpolation import shift

import numpy as np
import os
import sys

class CharacterPreparation(object):
    """
    Class used to manage access to system fonts for saving character images
    """

    font_dirs = ['/System/Library/Fonts/', '/Library/Fonts/']
    font_name_glob_pattern = '*.ttf'
    output_file = RAW_IMAGE_NAMING
    # Character ordering is maintained in each file
    chars_list = list(
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,./:;\'"\\[]{}()-_+=`~!@#$%^&*?'
    )
    char_count = len(chars_list)
    image_width = 32

    default_fontsize = 12

    def __init__(self, output_root=RAW_IMAGE_DIRECTORY):
        self.font_files = []
        for font_dir in self.font_dirs:
            self.font_files += glob(os.path.join(
                font_dir, self.font_name_glob_pattern
            ))
        self.color_text = 'black'
        self.color_background = 'white'

        self.arange = np.arange(1, self.image_width + 1)

        self.complete_out_path = os.path.join(
            output_root,
            self.output_file
        )

    def _get_array_com(self, arr):
        """ Who says a physics undergrad ed isn't useful for data science?
        Returns index of character center of mass.

        Returns: x_mid, y_mid

        Old way:
        normalization_const = arr.sum(axis=0).sum()
        x_com = 32 * np.mean(self.arange * arr.sum(axis=0)) / normalization_const
        y_com = 32 * np.mean(self.arange * arr.sum(axis=1)) / normalization_const
        """
        y_com, x_com = center_of_mass(arr)
        try:
            y_com = int(y_com)
            x_com = int(x_com)
        except:
            y_com, x_com = 16, 16
        return (x_com, y_com)

    def _center_array(self, arr, x_com, y_com):
        """Moves pixels to center them on the center of mass (COM)

        Old way:
        # Ensure pixels aren't rolling over to other side!
        def move(movement, arr, direction):
            if direction == 'x':
                if movement > 0:
                    # Left pad
                    return np.pad(arr, ((0,0), (movement, 0)), mode='constant')[:, :-movement]
                elif movement < 0:
                    # right pad
                    return np.pad(arr, ((0,0), (0, -movement)), mode='constant')[:, abs(movement):]
                else:
                    return arr
            elif direction == 'y':
                if movement > 0:
                    #  pad
                    return np.pad(arr, ((movement,0), (0, 0)), mode='constant')[:-movement, :]
                elif movement < 0:
                    # right pad
                    return np.pad(arr, ((0, -movement), (0, 0)), mode='constant')[abs(movement):, :]
                else:
                    return arr

        midpoint = self.image_width // 2
        move_x = midpoint-x_com
        move_y = midpoint-y_com
        print "Moving:", move_x, move_y

        arr = move(move_x, arr, 'x')
        arr = move(move_y, arr, 'y')
        """
        x_adj = int(self.image_width//2 - x_com)
        y_adj = int(self.image_width//2 - y_com)
        return shift(arr, (x_adj, y_adj))

    def _get_char_image(self, char, font_file):
        """Given char and font object return a 1 dimensional folded array
        with pixel values"""
        def optimize_fontsize(font_file, char):
            font = ImageFont.truetype(font_file, self.default_fontsize)
            (width, height) = font.getsize(char)
            x_factor, y_factor = self.image_width // width, self.image_width // height
            font_size_mult_factor = min(x_factor, y_factor)
            return font_size_mult_factor

        fontsize = optimize_fontsize(font_file, char) * self.default_fontsize
        # print "Selected fontsize =", fontsize
        font = ImageFont.truetype(
            font_file,
            fontsize
        )
        (width, height) = font.getsize(char)

        img = Image.new('L', (self.image_width, self.image_width), self.color_text)
        d = ImageDraw.Draw(img)
        d.text(
            (
                (self.image_width-width)/2, (self.image_width-height)/2
            ),
            char,
            fill=self.color_background,
            font=font
        )
        char_arr = np.array(img)
        x_com, y_com = self._get_array_com(char_arr)
        char_arr = self._center_array(char_arr, y_com, x_com)
        return Image.fromarray(char_arr, 'L')

    def _get_filename_kwargs(self, char, font_file):
        # If chaning keys, verify equality with class attribute "output_file"
        font_text = os.path.basename(font_file).split('.')[0]
        # Character are stored in directories after their hex because some
        # characters are not allowed to appear in directory names like '/'
        return {
            'hex_char_code': char.encode('hex'),
            'font': font_text
        }

    def _save_img(self, img, char, font_file):
        output_path = self.complete_out_path.format(
            **self._get_filename_kwargs(char, font_file)
        )
        directory = os.path.dirname(output_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        img.save(output_path)

    def write_font_chars(self):
        """
        Writes a csv file for each font image_width^2 columns corresponding to

        """
        for font_file in self.font_files:
            if font_file.split('/')[-1] not in goodFonts:
                print "Passing", font_file.split('/')[-1]
                continue
            else:
                print "Saving", font_file.split('/')[-1]

            for i, char in enumerate(self.chars_list):
                # import pdb; pdb.set_trace()
                img = self._get_char_image(char, font_file)
                self._save_img(img, char, font_file)

    def _apply_random_rotation(self, arr):
        degrees = np.random.uniform(0.0, 360.0)
        return rotate(arr, degrees, reshape=False)

if __name__ == '__main__':
    chSave = CharacterPreparation()

    # Generate base images and save them in directories
    # organized by character in hex format and named by font.
    chSave.write_font_chars()
