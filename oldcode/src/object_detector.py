import logging          
import cv2              
import os               
import numpy as np     
import sys              
import zerorpc          
import msgpack          
import msgpack_numpy as m 
import tensorflow as tf
from skimage.feature import match_template
import skimage
import skimage.io as io
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from copy import deepcopy

logger = logging.getLogger(__name__)


def draw_circle(event, x, y, flags, param):
    """
    Opencv mouse click event which draws a blue circle after a double click
    """
    if event == cv2.EVENT_LBUTTONDBLCLK:
        logger.debug('mouse click at (width={0},height={1})'.format(x, y))
        image = param.get('image')
        height, width = image.shape[0], image.shape[1]
        centers = param.get('centers')
        window_name = param.get('window_name')
        cv2.circle(image, (x, y), 10, (255, 0, 0), -1)
        centers.append((x / width, y / height))
        logger.debug(
            'relative center is (width={0},height={1})'.format(x / width, y / height))
        cv2.imshow(window_name, image)



'''
    def predict(self, image_pack):
        """
        Detect robots and return an object list.

        Parameters
        ----------
        image: numpy array
            an image consists of the gameing board and multiple robots

        Returns
        -------
        object_list: dict
            objects' relative locations, relative sizes and categories
            example:

        """
        image = msgpack.unpackb(image_pack, object_hook=m.decode)
        im = self.convert_image(image)
        results = dn.detect_image(self.net, self.meta, im)
        results = [[result[0].decode('utf-8'), result[1], result[2]] for result in results]
        return results

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

'''
class Detector:
    """
    Object detector based on YOLO network.
    """

    def __init__(self, weight_path, network_config_path=None, 
                 object_config_path=None, auto_id=False, remote=None):
        """
        Load YOLO network weights and config files

        Parameters
        ----------
        weight_path: str
            file path of YOLOv3 network weights
        network_config_path: str
            file path of YOLOv3 network configurations
        object_config_path: str
            file path of object configurations
        """
        self.future_keys = { # future : past
                'cheese': 'thief',
                'pink' : 'policeman1',
                'blue' : 'policeman2'}
        self.object_list_template = {
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

        if remote is None:
            input_shape_tuple = (64, 64, 3)
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
            self.nn = model
        #else:
            #self.dark = zerorpc.Client(heartbeat=None)
            # address = 'tcp://{ip}:{port}'.format(ip=self.ip, port=self.port)
            #self.dark.connect(remote)

        # self.width = self.dark.get_width()
        # self.height = self.dark.get_height()
        self.policemen = {}
        self.auto_id = auto_id
        self.fake_id = 0

    def detect_objects(self, input_img, display=False, file=None):
        """
        Detect robots and return an object list.

        Parameters
        ----------
        image: numpy array
            an image consists of the gameing board and multiple robots

        Returns
        -------
        object_list: dict
            objects' relative locations, relative sizes and categories
            example:

        """
            
        img2 = input_img.copy()
        match_these = {'pink': io.imread(r'/home/mikhail/Desktop/Robotics/oldcode/data/templates/pink_rat.jpg'),
                       'blue': io.imread(r'/home/mikhail/Desktop/Robotics/oldcode/data/templates/blue_rat.jpg'),
                       'cheese': io.imread(r'/home/mikhail/Desktop/Robotics/oldcode/data/templates/cheese.jpg')}

        # matching_methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED']
        # matching_methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
        #                     'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
        matching_methods = ['skimage.feature.match_template']
        object_list = deepcopy(self.object_list_template)

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
                nn_out = self.nn.predict(resized_guess, steps=1)
                method_scores[met]['score'] = nn_out[0][nn_output_legend[key]]

                template_key = self.future_keys[key]
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
        
        logger.debug('object list: {}'.format(object_list))
        if len(object_list) < 3:
            logger.warning(
                'Only {} objects are recognized'.format(len(object_list)))
        return object_list

    def track_objects(self, detect_results):
        policemen = {}
        object_list = {}
        for result in detect_results:
            name = result[0]
            confidence = result[1]
            bounds = result[2]
            center = (bounds[0] / self.width, bounds[1] / self.height)
            size = (bounds[2] / self.width, bounds[3] / self.height)
            if name == 'policeman' and len(self.policemen) == 0:
                if self.auto_id:
                    police_id = self.fake_id
                    self.fake_id += 1
                else:
                    police_id = input("Please input a police id for object at {}".format(center))
                name = '{0}{1}'.format('policeman', police_id)
                policemen[name] = {
                    'confidence': confidence,
                    'center': center,
                    'size': size
                }
            elif name == 'policeman' and len(self.policemen) != 0:
                name = self.get_police_name_by_distance(center)
                policemen[name] = {
                    'confidence': confidence,
                    'center': center,
                    'size': size
                }
            object_list[name] = {
                'confidence': confidence,
                'center': center,
                'size': size
            }
        for key, value in policemen.items():
            self.policemen[key] = value
        for name, policeman in policemen.items():
            object_list[name] = policeman
        return object_list

    def get_police_name_by_distance(self, center):
        names = []
        distances = []
        for police_name, police in self.policemen.items():
            names.append(police_name)
            current_center = np.array(police['center']).reshape((-1, 1))
            future_center = np.array(center).reshape((-1, 1))
            distances.append(np.linalg.norm(future_center - current_center))
        index = np.argmin(distances)
        name = names[int(index)]
        return name

    def detect_gaming_board(self, image):
        """
        Analysis the gaming board image to obtain centers of triangles.

        Parameters
        ----------
        image: numpy array
            an image consists of the gaming board(may not contains robots)

        Returns
        -------
        centers: list
            relative coordinates of triangles on the gaming board(width,height)
        """
        # RBG image to BGR image for better visualization
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # build a Opencv window and set up a mouse click event
        centers = []
        window_name = 'center_tool'
        cv2.namedWindow(window_name)
        callback_params = {
            'image': frame,
            'centers': centers,
            'window_name': window_name
        }
        cv2.setMouseCallback(window_name, draw_circle, param=callback_params)
        cv2.imshow(window_name, frame)

        # wait for exit flag
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyWindow(window_name)

        # save or read centers file
        center_file_path = 'centers.txt'
        if not centers:
            assert os.path.exists(center_file_path)
            with open(center_file_path, encoding='utf-8', mode='r') as file:
                for line in file:
                    center = tuple(map(float, line.strip().split(' ')))
                    centers.append(center)
        else:
            with open(center_file_path, encoding='utf-8', mode='w') as file:
                for center in centers:
                    file.write('{width} {height}\n'.format(
                        width=center[0], height=center[1]))
        return centers
'''
    def convert_image(self, image):
        """
        Convert numpy array to specific format which can be used by darknet library.

        Parameters
        ----------
        image: numpy array
            a three-dimensional array with uint type

        Returns
        -------
        im: custom object
            an object which is defined by darknet library
        """
        resized_image = cv2.resize(
            image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        im, _ = dn.array_to_image(resized_image)
        return im
'''

if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    from camera_system import Camera

    weight_path = '../model/custom_tiny_yolov3.weights'
    network_config_path = '../cfg/custom-tiny.cfg'
    object_config_path = '../cfg/custom.data'
    detector = Detector(weight_path, network_config_path, object_config_path, auto_id=True)
    camera = Camera(1, save=False, draw=False, num_skip=0)

    window_name = 'test'
    cv2.namedWindow(window_name)

    while True:
        image = camera.get_image()
        if image is None:
            break
        image = camera.rgb_to_bgr(image)
        object_list = detector.detect_objects(image)
        boxes = camera.draw_boxes(image, object_list)
        cv2.imshow(window_name, boxes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow(window_name)
