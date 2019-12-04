'''
    Import OpenCV package
    Import universally unique identifier package
    Import standard library within Python
'''

import cv2
import uuid
import os

# These are all just drawing functions for OpenCV
'''
    Assigns colors
    thief - Red
    policeman1 - Green
    policeman2 - Blue
    not sure why yet
'''
COLORS = {
    'thief': (255, 0, 0),
    'policeman1': (0, 255, 0),
    'policeman2': (0, 0, 255)
}
# Sets normal size sans-serif font
FONT = cv2.FONT_HERSHEY_SIMPLEX
# Separate scaling factor to independently scale the size of the font elements
FONT_SCALE = 1
# Line type. Don't know what "2" line type means, but guessing it could be a filled line
LINE_TYPE = 2

# Class containing a bunch of functions
class Camera:
    @staticmethod
    # Function that reads an image of the gaming board from a path and converts it to a format used in OpenCV
    def get_fake_gaming_board():
        # Reads the image from the path
        frame = cv2.imread('../resources/gaming_board.jpg')
        # Converts from RGB to BGR format for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Returns the image of the gaming board
        return frame
    
    # Function that capures videos from the camera using its specific id and creates a window for them
    def __init__(self, camera_id=0, draw=True, save=True, save_path='../data/pics', num_skip=10, window_name='main'):
        self.draw = draw
        self.save = save
        self.save_path = save_path
        # Captures videos from the camera
        self.cap = cv2.VideoCapture(camera_id)
        self.num_skip = num_skip
        self.window_name = window_name
        if self.draw:
            # Creates a window that is used as a placeholder for images
            cv2.namedWindow(window_name)

    # Function that closes the window
    def __del__(self):
        self.cap.release()
        if self.draw:
            # Closes the window
            cv2.destroyWindow(self.window_name)

    # Honestly have no clue
    def _skip_frames(self):
        for i in range(self.num_skip):
            self.cap.read()

    # Function looks for an image, saves it, and returns it. If there's no image, return nothing.
    def get_image(self):
        self._skip_frames()
        flag, frame = self.cap.read()
        if flag:
            if self.save:
                # Saves an image to a specified file
                cv2.imwrite(os.path.join(self.save_path, '{}.jpg'.format(uuid.uuid1())), frame)
            if self.draw:
                self.display(frame)
            # Converts to BGR format for OpenCV
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            image = None
        return image

    # Function that draws boxes around the thief, policeman1, and policeman2, from the center of the game board.
    # This looks like it's based off the drawn image of the game board, not actual images taken off the camera.
    @staticmethod
    def draw_boxes(image, object_list):
        if len(object_list) > 0:
            for key, value in object_list.items():
                height, width = image.shape[0], image.shape[1]
                x = int(value['center'][0] * width)
                y = int(value['center'][1] * height)
                size_width = value['size'][0] * width
                size_height = value['size'][1] * height
                x1 = int(x - size_width / 2)
                y1 = int(y - size_height / 2)
                x2 = int(x + size_width / 2)
                y2 = int(y + size_height / 2)
                color = COLORS.get(key, (255, 255, 255))
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, key,
                            (x1 - 10, y1 - 10),
                            FONT,
                            FONT_SCALE,
                            color,
                            LINE_TYPE)
        # Returns the image containing the drawn boxes
        return image

    # Function that displays each image for a short period in a window
    def display(self, image):
        # Displays an image in a window that auto fits to the image size
        cv2.imshow(self.window_name, image)
        # Displays a frame for 1 ms and waits for a key press
        cv2.waitKey(1)

    # Function that converts an image from (blue, green, red) to (red, green, blue) and returns it
    @staticmethod
    def bgr_to_rgb(frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return image

    # Function that converts an image from (red, green, blue) to (blue, green, red) and returns it
    @staticmethod
    def rgb_to_bgr(frame):
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return image

# This will run when file is run, unless file is imported elsewhere
if __name__ == '__main__':
    camera = Camera(0, save=False, num_skip=0)
    while True:
        # Gets image
        image = camera.get_image()
        # Converts image
        image = camera.rgb_to_bgr(image)
        # Displays image
        camera.display(image)
