from skimage import data
from skimage.feature import match_template
import skimage
import numpy as np
import skimage.io as io
from matplotlib import pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from copy import deepcopy
import h5py
weight_path = r'D:\python\mikhailAssets\2019-12-07_23-22-17_trainedmodel.h5'
localDrive = 'D:'
os.chdir(localDrive)
print(os.getcwd())
future_keys = { # future : past
                'cheese': 'thief',
                'pink' : 'policeman1',
                'blue' : 'policeman2'}
object_list_template = {
    "thief": {          # cheese
        "confidence": 0.99,
        "center": (0.5, 0.5),  # (width,height)
        "size": (0.15, 0.10),  # (width,height)
    },
    "policeman1": {     # white mouse
        "confidence": 0.99,
        "center": (0.5, 0.5),  # (width,height)
        "size": (0.15, 0.05),  # (width,height)
    },
    "policeman2": {     # blue mouse
        "confidence": 0.99,
        "center": (0.5, 0.5),  # (width,height)
        "size": (0.15, 0.05),  # (width,height)
    }
}
input_shape_tuple = (64, 64, 3)
f = h5py.File(weight_path, 'r')
model = tf.keras.models.Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=input_shape_tuple))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3)) # convergence layer
model.add(Activation('softmax'))

model.load_weights(weight_path)

def detect_objects(input_img, display=False, file=None):
    """
    Detect robots and return an object list.

    Parameters
    ----------
    input_img: numpy array
        an image consists of the gameing board and multiple robots

    Returns
    -------
    object_list: dict
        objects' relative locations, relative sizes and categories
        example:

    """
    img2 = input_img.copy()
    match_these = {'pink': io.imread(r'D:\python\mikhailAssets\templates\pink\wm_e_pi_j.jpg'),
                   'blue': io.imread(r'D:\python\mikhailAssets\templates\blue\bm_e_pi_j.jpg'),
                   'cheese': io.imread(r'D:\python\mikhailAssets\templates\cheese\c_e_pi_j.jpg')}

    # matching_methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED']
    # matching_methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
    #                     'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    matching_methods = ['skimage.feature.match_template']
    object_list = deepcopy(object_list_template)

    for key in match_these.keys():
        nn_output_legend = {'blue': 0,
                            'cheese': 1,
                            'pink': 2}
        # Contains scores from the neural network for the template matcher's guess
        # We will pack this array with scores for each method and use da best
        method_scores = {'skimage.feature.match_template': {'score': 0, 'center': (0, 0), 'size': (0, 0)},
                         'cv2.TM_CCOEFF': {'score': 0, 'center': (0, 0), 'size': (0, 0)},
                         'cv2.TM_CCOEFF_NORMED': {'score': 0, 'center': (0, 0), 'size': (0, 0)},
                         'cv2.TM_CCORR': {'score': 0, 'center': (0, 0), 'size': (0, 0)},
                         'cv2.TM_CCORR_NORMED': {'score': 0, 'center': (0, 0), 'size': (0, 0)},
                         'cv2.TM_SQDIFF': {'score': 0, 'center': (0, 0), 'size': (0, 0)},
                         'cv2.TM_SQDIFF_NORMED': {'score': 0, 'center': (0, 0), 'size': (0, 0)}}
        for met in matching_methods:
            # met = 'cv2.TM_CCOEFF_NORMED'
            # if key == 'cheese':
            #     met =  'cv2.TM_CCOEFF'
            img = img2.copy()
            # if key == 'blue':
            #     image_colorfulness(img)

            result = match_template(img, match_these[key])
            ij = np.unravel_index(np.argmax(result), result.shape)
            x, y = ij[0], ij[1]
            w, h = match_these[key].shape[0], match_these[key].shape[1]
            # Call the neural network to check validity
            guess = img2[w:x, h:y]
            # viewer = skimage.viewer.ImageViewer(guess)
            # viewer.show()

            #
            # cv2.imwrite(r'D:\python\mikhailAssets\image_under_test.jpg', guess)
            # image_under_test = [tf.image.decode_jpeg(r'D:\python\mikhailAssets\image_under_test.jpg', channels=0)]
            resized_guess = skimage.transform.resize(guess, (1, 64, 64, 3))
            nn_out = model.predict(resized_guess, steps=1)
            method_scores[met]['score'] = nn_out[0][nn_output_legend[key]]

            template_key = future_keys[key]
            method_scores[met]["center"] = ((x + w) / 2, (y + h) / 2)
            method_scores[met]["size"] = (w, h)
        # for idx, met_key in enumerate(method_scores.keys()):
        #     best_score = 0
        #     best_idx = 0
        #     best_method = 'str'
        #     if method_scores[met_key]['score'] > best_score:
        #         best_method = met_key

        object_list[template_key]["center"] = ((x + w) / 2, (y + h) / 2)
        object_list[template_key]["confidence"] = nn_out[0][nn_output_legend[key]]
        object_list[template_key]["size"] = (w, h)
        #
        # object_list[template_key]["center"] = method_scores[best_method]['center']
        # object_list[template_key]["size"] = method_scores[best_method]['size']


        if display:
            print(object_list)

            fig = plt.figure(figsize=(8, 3))
            ax1 = plt.subplot(1, 3, 1)
            ax2 = plt.subplot(1, 3, 2)
            ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)

            ax1.imshow(guess, cmap=plt.cm.gray)
            ax1.set_axis_off()
            ax1.set_title('template')

            ax2.imshow(input_img, cmap=plt.cm.gray)
            ax2.set_axis_off()
            ax2.set_title('image')
            # highlight matched region
            rect = plt.Rectangle((x, y), w, h, edgecolor='r', facecolor='none')
            ax2.add_patch(rect)

            ax3.imshow(result)
            ax3.set_axis_off()
            ax3.set_title('`match_template`\nresult')
            # highlight matched region
            ax3.autoscale(False)
            ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

            plt.show()
    if file is not None:
        print(file)
        print(object_list)
    return object_list

if __name__ == '__main__':

    img_dir =  r'D:\python\mikhailAssets\robot_pix'
    that_dir = os.listdir(img_dir)
    for file in that_dir:
        if file[-4:] == ".jpg" or file[-4:] == ".png":
            try:
                img = io.imread(os.path.join(img_dir, file))
                detect_objects(img, display=False, file=file)
            except:
                pass
