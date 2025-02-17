from __future__ import print_function
from camera_system import Camera
from object_detector import Detector
from strategy import Strategy
from graph_builder import GraphBuilder
from control_system import Controller
import logging
import sys
import time
import json
import random
import os
# videocapture_path = r'/home/mikhail/Downloads/VideoCapture-0.9-5/Python27/DLLs_64/'
# os.environ['path'] = videocapture_path + os.pathsep + os.environ['PATH']

WEIGHT_PATH = '../model/custom_tiny_yolov3.weights'
NETWORK_CONFIG_PATH = '../cfg/custom-tiny.cfg'
OBJECT_CONFIG_PATH = '../cfg/custom.data'
ROBOTS_CONFIG_PATH = '../cfg/robots.json'

logger = logging.getLogger(__name__)


class FakeGame:
    def __init__(self):
        self.camera = Camera(None, draw=False)
        #self.display_camera = Camera(None, window_name='labeled')
        centers = []
        with open('centers.txt', encoding='utf-8', mode='r') as file:
            for line in file:
                center = tuple(map(float, line.strip().split(' ')))
                centers.append(center)
        self.centers = centers
        self.graph_builder = GraphBuilder(self.centers)
        self.orders = ['thief', 'policeman1', 'policeman2']
        self.strategy = Strategy(self.orders)
        self.object_list = {
            "thief": {
                "confidence": 0.99,
                "center": self.centers[6],  # (width,height)
                "size": (0.15, 0.10),  # (width,height)
            },
            "policeman1": {
                "confidence": 0.99,
                "center": self.centers[1],  # (width,height)
                "size": (0.15, 0.05),  # (width,height)
            },
            "policeman2": {
                "confidence": 0.99,
                "center": self.centers[3],  # (width,height)
                "size": (0.15, 0.05),  # (width,height)
            }
        }
        self.counter = 0
        self.thief_movements = [13, 14, 15, 16]
        self.escape_nodes = {10}
        self.graph = None
        self.objects_on_graph = None
        self.instructions = None

    def forward(self):

        image = self.camera.get_fake_gaming_board()
        self.display_camera.draw_boxes(image, self.object_list)
        self.display_camera.display(image)

        # build a graph based on object list
        graph, objects_on_graph = self.graph_builder.build(self.object_list)

        self.graph = graph
        self.objects_on_graph = objects_on_graph

        # generate instructions based on the graph
        instructions = self.strategy.get_next_steps_shortest_path(graph, objects_on_graph)
        logger.info('instructions:{}'.format(instructions))

        # instructions['thief'] = [objects_on_graph['thief'], self.thief_movements[self.counter]]
        self.instructions = instructions

        self.counter += 1
        for key, value in instructions.items():
            self.object_list[key]['center'] = self.centers[value[1] - 1]
        time.sleep(1)

        image = self.camera.get_fake_gaming_board()
        self.display_camera.draw_boxes(image, self.object_list)
        self.display_camera.display(image)

    def is_over(self):
        """
        Check if the game is over.

        Returns
        -------
        game_over: bool
            True if the thief is at the escape point or the policemen have caught the thief, otherwise False.
        """
        game_over = False
        if self.instructions is None or self.objects_on_graph is None or self.graph is None:
            return game_over
        if 'thief' in self.objects_on_graph:
            if self.objects_on_graph['thief'] in self.escape_nodes:
                game_over = True
                logger.info('The thief wins!')
            else:
                for name, instruction in self.instructions.items():
                    if name != 'thief':
                        if self.instructions['thief'][1] == instruction[1]:
                            game_over = True
                            logger.info('The policemen win!')
        return game_over

    def get_report(self):
        """
        Generate a game report(json, xml or plain text).

        Returns
        -------
        game_report: object or str
            a detailed record of the game
        """
        game_report = None
        return game_report

    def shuffle(self):
        random.randint(5, 10)


class Game:
    """
    Each game is an instance of class Game.
    """

    def __init__(self, weight_path, robots_config_path, network_config_path=None, object_config_path=None):
        """
        Load necessary modules and files.

        Parameters
        ----------
        weight_path: str
            file path of YOLOv3 network weights
        network_config_path: str
            file path of YOLOv3 network configurations
        object_config_path: str
            file path of object information in YOLOv3 network
        robots_config_path: str
            file path of robots' remote server configuration
        """

        # fix robot movement order
        self.orders = ['thief', 'policeman1']
        # self.orders = ['policeman1', 'policeman2']
        # self.orders = ['thief', 'policeman1', 'policeman2']

        # initialize internal states
        self.graph = None
        self.objects_on_graph = None
        self.instructions = None

        # set up escape nodes
        self.escape_nodes = set()

        # construct the camera system
        self.camera = Camera(1)

        # construct the object detector
        self.detector = Detector(weight_path)

        # load gaming board image and get centers' coordinates of triangles
        self.gaming_board_image = self.camera.cap.getImage()
        self.centers = self.detector.detect_gaming_board(self.gaming_board_image)

        # construct the graph builder
        self.graph_builder = GraphBuilder(self.centers)

        # construct the strategy module
        self.strategy = Strategy(self.orders)

        # construct the control system
        self.controller = Controller(self.detector, self.camera.get_image, robots_config_path)

        # connect to each robot
        self.controller.connect()

    def is_over(self):
        """
        Check if the game is over.

        Returns
        -------
        game_over: bool
            True if the thief is at the escape point or the policemen have caught the thief, otherwise False.
        """
        game_over = False
        if self.instructions is None or self.objects_on_graph is None or self.graph is None:
            return game_over
        if 'thief' in self.objects_on_graph:
            if self.objects_on_graph['thief'] in self.escape_nodes:
                game_over = True
                logger.info('The thief wins!')
            else:
                for name, instruction in self.instructions.items():
                    if name != 'thief':
                        if self.instructions['thief'][1] == instruction[1]:
                            game_over = True
                            logger.info('The policemen win!')
        return game_over

    def shuffle(self):
        random.randint(5, 10)

    def forward(self):
        """
        Push the game to the next step.
        """
        # get objects' coordinates and categories
        image = self.camera.get_image()
        object_list = self.detector.detect_objects(image)

        # build a graph based on object list
        graph, objects_on_graph = self.graph_builder.build(object_list)
        self.graph = graph
        self.objects_on_graph = objects_on_graph

        # generate instructions based on the graph
        instructions = self.strategy.get_next_steps_shortest_path(graph, objects_on_graph)
        self.instructions = instructions
        logger.info('instructions:{}'.format(instructions))

        if self.is_over():
            return
        # move robots until they reach the right positions
        while not self.controller.is_finished(self.centers, object_list, instructions):
            # obtain feedback from camera
            image = self.camera.get_image()
            object_list = self.detector.detect_objects(image)

            # calculate control signals
            control_signals = self.controller.calculate_control_signals(
                self.centers, object_list, instructions)

            # cut extra signals
            real_signals = []
            for name in self.orders:
                for signal in control_signals:
                    if signal['name'] == name:
                        # if True:
                        real_signals.append(signal)
                if len(real_signals) > 0:
                    break

            # update internal states
            self.controller.update_state(object_list)

            # move robots
            self.controller.move_robots(real_signals)

            # obtain feedback from camera
            image = self.camera.get_image()
            object_list = self.detector.detect_objects(image)

            # update internal states
            self.controller.update_state(object_list)

    def get_report(self):
        """
        Generate a game report(json, xml or plain text).

        Returns
        -------
        game_report: object or str
            a detailed record of the game
        """
        game_report = None
        return game_report


def main():
    # set up logger level
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(r'/home/mikhail/Desktop/Robotics/oldcode/badrun_log.log')
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    # parse config file
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = '../cfg/game_config.json'
    with open(config_path, encoding='utf-8', mode='r') as file:
        config = json.load(file)

    # load game parameters
    #weight_path = config['weight_path']
    weight_path = r'/home/mikhail/Desktop/Robotics/oldcode/data/2019-12-07_23-22-17_trainedmodel.h5'
    #network_config_path = config['network_config_path']
    #object_config_path = config['object_config_path']
    robots_config_path = config['robots_config_path']

    # construct a game logic
    game = Game(weight_path, robots_config_path)
    # game = FakeGame()
    # start the game logic
    while True:
        input('Press ENTER to the start a game:')

        # keep running until game is over
        while not game.is_over():
            game.forward()

        # get the game report
        report = game.get_report()

        # display the game report
        print(report)

        # shuffle the robots on the gaming board
        # TODO: finish shuffle() function
        game.shuffle()


if __name__ == '__main__':
    main()
