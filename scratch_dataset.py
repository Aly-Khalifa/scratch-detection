import glob
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os.path


class ScratchDataset:

    def __init__(self):
        self.num_images = 0
        self.image_height = 0
        self.image_width = 0
        self.image_bit_depth = 8
        self.frames = None
        self.masks = None
        self.frames_path = 'npdata/frames.npy'
        self.masks_path = 'npdata/masks.npy'

        self.load_dataset()

    def load_dataset(self):
        if not os.path.isfile(self.masks_path):
            self.load_dataset_from_files()
            return

        self.frames = np.load(self.frames_path)
        self.masks = np.load(self.masks_path)
        self.num_images = self.frames.shape[0]
        self.image_height, self.image_width, _ = self.frames[0].shape

    # loads the dataset
    def load_dataset_from_files(self):
        # load paths to frames and masks
        frame_paths = glob.glob('./frames/*.png');
        mask_paths = [frame_path.replace('frames/mlts', 'masks/label') for frame_path in frame_paths]
        assert (len(frame_paths) == len(mask_paths)), "The number of frames and masks did not match"

        # determine image dimensions
        self.image_height, self.image_width, _ = img_to_array(load_img(frame_paths[0])).shape

        n_imgs = len(frame_paths)
        self.num_images = n_imgs

        x = np.empty(shape=(n_imgs, self.image_height, self.image_width, 1), dtype='float32')
        y = np.empty(shape=(n_imgs, self.image_height, self.image_width, 1), dtype='float32')

        for i in range(n_imgs):
            # load the frame from file
            frame = img_to_array(load_img(frame_paths[i], color_mode="grayscale")).astype('float32')
            x[i, :, :, :] = frame

            # load the corresponding mask
            mask = img_to_array(load_img(mask_paths[i], color_mode="grayscale")).astype(('float32'))
            mask[mask > 0] = 1  # set class label to one
            y[i, :, :, :] = mask

        self.frames = x
        self.masks = y

        np.save('npdata/frames.npy', x)
        np.save('npdata/masks.npy', y)

        assert x.shape == np.load(self.frames_path).shape, "Saved numpy file does not match the parsed data"

    def get_patches(self, number, size, train_size=None):
        # check that data has been loaded
        assert self.frames is not None, 'frames must be loaded using the load_dataset() method'
        assert self.masks is not None, 'masks must be loaded using the load_dataset() method'

        if train_size is None:
            return ScratchDataset.__generate_patches(self.frames, self.masks, number, size)
        else:
            frames_train, frames_test, masks_train, masks_test = \
                train_test_split(self.frames, self.masks, train_size=train_size)

            n_train = round(train_size * number)
            n_test = round((1 - train_size) * number)

            frame_patches_train, mask_patches_train = ScratchDataset.__generate_patches(frames_train, masks_train,
                                                                                        n_train, size)
            frame_patches_test, mask_patches_test = ScratchDataset.__generate_patches(frames_test, masks_test, n_test,
                                                                                      size)

            return frame_patches_train, frame_patches_test, mask_patches_train, mask_patches_test

    def display_random_patches(self, size):

        frames, masks = self.get_patches(9, size)

        fig, axes = plt.subplots(3, 6)
        axes = np.reshape(axes, (9, 2))
        fig.suptitle("Random Sample of Generated Patches")

        for i, frame, mask, ax in zip(range(9), frames, masks, axes):
            # display patches
            ax[0].imshow(np.squeeze(frame))
            ax[0].set_title("{}".format(i))

            ax[1].imshow(np.squeeze(mask))
            ax[1].set_title("{}".format(i))

        plt.show()

    @staticmethod
    def __generate_patches(frames: np.ndarray, masks: np.ndarray, number, size, sparse_limit=0.5):

        num_images, image_height, image_width, _ = frames.shape

        sparse_limit_num = round(sparse_limit * num_images)

        # initialise variables to store patches
        frame_patches = np.empty(shape=(number, size, size, 1))
        mask_patches = np.empty(shape=(number, size, size, 1))

        num_sparse = 0
        # select random images to obtain patches from
        for i, j in zip(range(number), np.random.randint(0, num_images, number)):

            # this is to prevent an infinite loop from happening
            if np.sum(masks[j]) == 0:
                continue

            while True:
                # randomly select a coordinate within the image for patch extraction
                row = int(np.random.randint(0, image_height - size, 1))
                col = int(np.random.randint(0, image_width - size, 1))

                # get a random mask patch
                mask = masks[j, row:row + size, col:col + size, :]

                # if the mask is empty
                if np.sum(mask) == 0:
                    num_sparse += 1

                    # if the limit for empty
                    if num_sparse >= sparse_limit_num:
                        continue

                # slice the patch out of the image
                frame_patches[i, :, :, :] = frames[j, row:row + size, col:col + size, :]
                mask_patches[i, :, :, :] = mask
                break

        return frame_patches, mask_patches
