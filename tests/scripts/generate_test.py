'''Generate test images.'''

import os, sys
sys.path.append(os.path.abspath('../../'))
import Quartz, AppKit
from flowbot import FlowBot

TEST_ROOT = '../../tests/'
TEST_DIR = os.path.join(TEST_ROOT, 'standard_globular')
SAVE_PNG = 'scattered-150.png'

if __name__ == '__main__':
    window = FlowBot.get_window("Flow")
    puzzle_img = FlowBot.screen_capture(window, save_png=os.path.join(TEST_DIR, SAVE_PNG))