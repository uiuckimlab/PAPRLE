LEADERS_DICT = {}

from paprle.leaders.sliders import Sliders
LEADERS_DICT['sliders'] = Sliders

try:
    from paprle.leaders.puppeteer import Puppeteer
    LEADERS_DICT['puppeteer'] = Puppeteer
except ImportError as e:
    print("Puppeteer not available. Please install the required dependencies to use it.")
    print(e)

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

try:
    from paprle.leaders.visionpro import VisionPro
    LEADERS_DICT['visionpro'] = VisionPro
except ImportError:
    #raise
    print("Visionpro not available. Please install the required dependencies to use it.")

try:
    from paprle.leaders.umi_dataset import UMIDataset
    LEADERS_DICT['umi'] = UMIDataset
except ImportError:
    print("UMIDataset not available. Please install the required dependencies to use it.")