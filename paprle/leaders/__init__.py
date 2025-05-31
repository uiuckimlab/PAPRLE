LEADERS_DICT = {}

from paprle.leaders.sliders import Sliders
LEADERS_DICT['sliders'] = Sliders

from paprle.leaders.puppeteer import Puppeteer
LEADERS_DICT['puppeteer'] = Puppeteer

from paprle.leaders.sim_puppeteer import SimPuppeteer
LEADERS_DICT['sim_puppeteer'] = SimPuppeteer

from paprle.leaders.keyboard import KeyboardController
LEADERS_DICT['keyboard'] = KeyboardController

try:
    from paprle.leaders.joycon import JoyconController
    LEADERS_DICT['joycon'] = JoyconController
except ImportError:
    print("JoyconController not available. Please install the required dependencies to use it.")

try:
    from paprle.leaders.dualsense import DualSense
    LEADERS_DICT['dualsense'] = DualSense
except ImportError:
    print("DualSenseController not available. Please install the required dependencies to use it.")

from paprle.leaders.visionpro import VisionPro
LEADERS_DICT['visionpro'] = VisionPro


