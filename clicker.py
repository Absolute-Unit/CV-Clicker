"""
CV Clicker

The 'not stupid' Auto Clicker.

A simple programm that searches a specified area on your desktop repeatedly for a
specific image, moves the cursor there and repeatedly simulates mouse clicks.

Can be used for clicker games. Use at your own risk.
"""

import threading
from operator import add
from time import sleep
import numpy as np
import cv2
from PIL import ImageGrab
from pynput.keyboard import Listener
from pynput.mouse import Button, Controller

# init mouse controller
MOUSE = Controller()

# TODO: Read these from a config file
# set which keys close the programm
EXIT_KEYS = ['q']

# set mouse config
CLICK_OFFSET = [-65, 40]
CLICK_PER_SECOND = 200

# set search config
SEARCH_RUNS_PER_SECOND = 1
SEARCH_ATTEMPTS_PER_RUN = 5

# set game resolution [width, height]
GAME_RESOLTUTION = [1920, 1080]

# set absolute game position [x,y] eg: [312, 186]
GAME_POSITION = [0,0]

# calculate coordinates of area to capture [left_x, top_y, right_x, bottom_y]
CAPTURE_AREA = list(map(add, GAME_POSITION + GAME_POSITION,
                        [0, 0] + GAME_RESOLTUTION))

# set target image to look for
TARGET_IMAGE = cv2.imread('src/images/target.jpg')
TARGET_IMAGE_WIDTH = TARGET_IMAGE.shape[0]
TARGET_IMAGE_HEIGHT = TARGET_IMAGE.shape[1]

# set exit flag TODO: non-global exit
EXIT_FLAG = 0


def get_screen_grab(bounding_box=None):
    """
    Take a screengrab and return it as an BGR array.

    Arguments:
        bounding_box (list of int): The area to capture.
            [left_x, top_y, right_x, bottom_y] (default is CAPTURE_AREA)

    Returns:
        screen_grab (array of int): Captured area as BGR array.
    """
    if bounding_box is None:
        bounding_box = CAPTURE_AREA
    raw_screen_grab = np.array(ImageGrab.grab(bbox=bounding_box))
    screen_grab = cv2.cvtColor(
        raw_screen_grab,
        cv2.COLOR_RGB2BGR
    )
    return screen_grab


def search_for_target(image, target=TARGET_IMAGE, n_runs=SEARCH_ATTEMPTS_PER_RUN, s_method=cv2.TM_CCOEFF_NORMED):
    """
    Search input for target image and return its top left coordinates.

    Arguments:
        image (array of int): The image to search in. Must be in BGR format.
        target (array of int): The target image to look for. Must be in BGR format. (default is TARGET_IMAGE)
        n_runs  (int): Number of searches per call (default is SEARCH_ATTEMPTS_PER_RUN).
        s_method (str): The used search method in object matching (default is cv2.TM_CCOEFF_NORMED)

    Returns:
        new_target (list of int): The new target coordinates.

    Notes:
        For more information on the search method see https://docs.opencv.org/2.4/modules/imgproc/doc/object_detection.html.
    """
    min_val = max_val = 0
    for search_count in range(n_runs):
        print('Searching...')
        # TODO: set best search method based on results
        search_result = cv2.matchTemplate(
            image, target, s_method)
        min_val, max_val = cv2.minMaxLoc(search_result)[0:2]
        # TODO: flexible target detection threshold based on results, > 1 seems to work for now
        if abs(min_val) + abs(max_val) > 1:
            print('Target found.')
            new_target = list(sum(cv2.minMaxLoc(search_result)[3:4], ()))
            return new_target
        if search_count == n_runs:
            print('Target not found.')
    return 0


def calculate_target_position(search_result, target_width=None, target_height=None):
    """
    Calculate center coordinate.

    The calculation is based on given coordinates and image width and height.
    It takes the input coordinates (top left) and goes halfway to the bottom right coordinates.

    Arguments:
        search_result (list of int):  The top left coordinates of the search results.
        target_width (int, optional): Width of the image we are looking for (default is TARGET_IMAGE_WIDTH).
        target_height (int, optonal): Height of the image we are looking for (default is TARGET_IMAGE_HEIGHT).

    Returns:
        calculated_target (list of int): The calculated coordinates.
    """
    if target_width is None:
        target_width = TARGET_IMAGE_WIDTH
    if target_height is None:
        target_height = TARGET_IMAGE_HEIGHT
    # print('Calculating target position.')
    top_left = search_result
    bottom_right = (top_left[0] + int(target_width),
                    top_left[1] + int(target_height))
    calculated_target = (np.mean([top_left, bottom_right], axis=0))
    return calculated_target


def move_cursor(target, offset=None, game_offset=None):
    """
    Move cursor to coordinates plus defined click offset while taking the game position into account.

    Arguments:
        target (list of int): The given input coordinates.
        offset (list of int): The offset to include. [x,y] (default is CLICK_OFFSET).
        game_offset (list of int): Additional offset to include. (default is GAME_POSITION)
    """
    if offset is None:
        offset = CLICK_OFFSET
    if game_offset is None:
        game_offset = GAME_POSITION
    new_target = np.array(target)
    new_target = new_target + offset + game_offset
    print('Moving mouse to ', new_target)
    MOUSE.position = new_target


def search_handler(search_speed=SEARCH_RUNS_PER_SECOND):
    """
    Handle the search process and moving the mouse.

    Arguments:
        search_speed (int): Dictates how many searches consequently cursor updates occur per second (default is SEARCH_RUNS_PER_SECOND).
    """
    current_image = get_screen_grab()
    current_result = search_for_target(current_image)
    if current_result != 0:
        current_target = calculate_target_position(current_result)
        cursor_target = current_target
        if any(MOUSE.position != cursor_target):
            print('Target updated.')
            move_cursor(cursor_target)
    else:
        print('No target found.')
    sleep(1/search_speed)


def click_handler(click_speed=CLICK_PER_SECOND):
    """
    Handle the mouse clicking.

    Arguments:
        click_speed (int): Clicks per second (default is CLICK_PER_SECOND).
    """
    MOUSE.release(Button.left)
    MOUSE.press(Button.left)
    sleep(1/click_speed)


def init_keyboard_listener():
    """
    Initialize keyboard listener.
    """
    def on_press(key):
        """
        Executed whenever a key is pressed.
        """
        global EXIT_FLAG
        try:
            # print('Alphanumeric key {0} pressed.'.format(key.char))
            if str(key)[1:-1] in EXIT_KEYS:
                EXIT_FLAG = 1
        except AttributeError:
            # print('Special key {0} pressed.'.format(key))
            # TODO: Get non alphanumeric keys working.
            if str(key) in EXIT_KEYS:
                EXIT_FLAG = 1
    listener = Listener(on_press=on_press)
    listener.start()


class SimpleThread(threading.Thread):
    # TODO: proper doc
    """
    Simple thread implementation.
    """

    def __init__(self, thread_id, name):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.name = name

    def run(self):
        print("Starting " + self.name)
        global EXIT_FLAG
        while EXIT_FLAG != 1:
            # TODO proper runtime implementation based on thread. (Thread Action Handler or something)
            if self.thread_id == 0:
                search_handler()
            elif self.thread_id == 1:
                click_handler()
        print("Exiting " + self.name)


def main():
    """
    Initialize and start threads.

    Only starts the thread if SEARCH_RUNS_PER_SECOND or CLICK_PER_SECOND is not set to zero respectivly.
    """
    init_keyboard_listener()
    # create new threads
    search_thread = SimpleThread(0, "Thread-Search")
    click_thread = SimpleThread(1, "Thread-Click")
    # start threads
    if SEARCH_RUNS_PER_SECOND != 0:
        search_thread.start()
    if CLICK_PER_SECOND != 0:
        click_thread.start()


if __name__ == "__main__":
    main()
