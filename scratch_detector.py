import datetime

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Input, concatenate, Lambda, Conv2DTranspose
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scratch_dataset import ScratchDataset
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from sklearn.model_selection import train_test_split
import io
import glob


@tf.function
def weighted_binary_crossentropy(y_true, y_pred):
    # get the fraction of pixels that are one


    return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, 10)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class ScratchDetector:

    def __init__(self):
        self.patch_size = 128
        self.model: Model = None

        self.get_model()

    # Uses the U-Net model architecture
    def get_model(self):
        img_input = Input(shape=(self.patch_size, self.patch_size, 1))

        s = Lambda(lambda x: x / 255.0)(img_input)

        c0 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
        c0 = Dropout(0.1)(c0)
        c0 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c0)
        p0 = MaxPooling2D((2, 2))(c0)

        c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p0)
        c1 = Dropout(0.1)(c1)
        c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = Dropout(0.1)(c2)
        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = Dropout(0.2)(c3)
        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = Dropout(0.2)(c4)
        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = Dropout(0.3)(c5)
        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4], axis=3)
        c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3], axis=3)
        c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2], axis=3)
        c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = Dropout(0.1)(c8)
        c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = Dropout(0.1)(c9)
        c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

        u10 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(c9)
        u10 = concatenate([u10, c0], axis=3)
        c10 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u10)
        c10 = Dropout(0.1)(c10)
        c10 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c10)

        #linear activation is used because the loss function used expects logit input.
        outputs = Conv2D(1, (1, 1), activation='linear')(c10)

        model = Model(inputs=[img_input], outputs=[outputs])

        model.summary()

        # compile the model
        model.compile(
            loss=weighted_binary_crossentropy,
            optimizer='adam',
            metrics=["accuracy"]
        )

        self.model = model

    def train_model(self, batch_size=128, num_epochs=100, num_patches=20000, train_size=0.8):
        if self.model is None:
            self.get_model()

        # load data
        data = ScratchDataset()
        data.load_dataset()

        frames_train, frames_test, masks_train, masks_test = data.get_patches(
            num_patches, self.patch_size, train_size=train_size)
        print("Loaded patches of sizes: ", frames_train.shape, frames_test.shape, masks_train.shape, masks_test.shape)

        # create logger
        logger = tf.keras.callbacks.TensorBoard(
            log_dir='logs/{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
            write_graph=True,
            histogram_freq=5
        )

        # Create a callback that saves the model's weights
        checkpoint_path = "checkpoints/{}/cp.ckpt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         verbose=1,
                                                         save_weights_only=True,
                                                         save_freq='epoch')

        # stop training early if accuracy target reached
        earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=0)

        self.model.fit(frames_train,
                       masks_train,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       verbose=2,
                       callbacks=[logger, cp_callback, earlystop_callback],
                       validation_data=(frames_test, masks_test)
                       )

        print("training complete")

    # loads the most recently trained model from the checkpoints folder
    def load_most_recent_model(self):

        if self.model is None:
            self.get_model()

        # get all folders containing checkpoints
        ckpt_folders = sorted(glob.glob("./checkpoints/*"))
        latest = tf.train.latest_checkpoint(ckpt_folders[-1])
        print("loading model weights from: ", latest)

        self.model.load_weights(latest)

    def predict_mask(self, frames):
        masks = self.model.predict(frames)

        # apply sigmoid function:
        masks = sigmoid(masks)

        return masks

    def display_predicted_mask(self, image_path, threshold=0.5):

        # load image from file
        x = np.squeeze(img_to_array(load_img(image_path, color_mode="grayscale")))

        # pad image to make it divisible by patch_size
        im_height, im_width = x.shape
        if im_height % self.patch_size == 0:
            pad_rows = 0
        else:
            pad_rows = self.patch_size - (im_height % self.patch_size)

        if im_width % self.patch_size == 0:
            pad_cols = 0
        else:
            pad_cols = self.patch_size - (im_width % self.patch_size)
        x_pad = np.pad(x, [(0, pad_rows), (0, pad_cols)], 'edge')

        # split image into patches
        patches = []
        for i in range(0, x_pad.shape[0], self.patch_size):
            for j in range(0, x_pad.shape[1], self.patch_size):
                patch = x_pad[i:i + self.patch_size, j:j + self.patch_size]
                patches.append(np.expand_dims(patch, axis=-1))

        patches = np.array(patches)
        print("patches shape: ", patches.shape)

        # make prediction using model
        masks = self.predict_mask(patches)

        # patch together the masks
        patch_indx = 0
        mask = np.expand_dims(np.empty(x_pad.shape), axis=-1)
        for i in range(0, x_pad.shape[0], self.patch_size):
            for j in range(0, x_pad.shape[1], self.patch_size):
                mask[i:i + self.patch_size, j:j + self.patch_size] = masks[patch_indx]
                patch_indx += 1

        # remove zero padding
        mask = mask[0: im_height, 0:im_width, :]
        print("final mask shape: ", mask.shape)

        # binarize mask
        if threshold is not None:
            mask[mask >= threshold] = 1
            mask[mask < threshold] = 0

        _, ax = plt.subplots(2, 1)
        #plot frame
        ax[0].imshow(x, cmap='gray')

        #plot mask blended with frame
        ax[1].imshow(x, cmap='gray')
        ax[1].imshow(np.squeeze(mask), cmap='cividis', alpha=0.5)

        #display image
        plt.show()

        return mask
