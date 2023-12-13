import time

import matplotlib.pyplot as plt
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

config = Cfg.load_config_from_name('vgg_transformer')
config['device'] = 'cpu'
detector = Predictor(config)

img = 'plate.jpg'
img = Image.open(img)
plt.figure()
plt.imshow(img)
t  = time.time()
s = detector.predict(img)


