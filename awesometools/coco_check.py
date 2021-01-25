import IPython
import os
import json
import random
import numpy as np
import requests
from io import BytesIO
import base64
from math import trunc
from PIL import Image as PILImage
from PIL import ImageDraw as PILImageDraw

import warnings

warnings.filterwarnings('ignore')

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

from pycocotools.coco import COCO

annotation_path = '/home/pzsuen/MoNuSAC/patches/val.json'
image_dir = '/home/pzsuen/MoNuSAC/patches/images'

coco = COCO(annotation_path)
print(coco.getCatIds())
print(len(coco.getImgIds()))
print(len(coco.getAnnIds()))
print(coco.getImgIds())
# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
print('\nCOCO categories: {}\n'.format(' '.join(nms)))

# select one at random
# select_imgId = int(coco.getImgIds()[random.randint(0, len(coco.getImgIds()))])
# print(select_imgId,type(select_imgId))
img_id = ['482', '1231']
coco_img = coco.getImgIds(img_id)
print(coco_img)
ci = coco.loadImgs(coco_img)
img = io.imread(os.path.join(image_dir, ci[1]['file_name']))

# annIds = coco.getAnnIds(imgIds=img_id, catIds=[1]) # 482
annIds = coco.getAnnIds(imgIds=img_id, catIds=[4]) # 1231

print(annIds)
print(len(annIds))

anns = coco.loadAnns(annIds)
plt.imshow(img)
coco.showAnns(anns)
plt.show()
# this_id = imgIds[0]
#
# ids = coco.getImgIds(this_id)
# print(ids)
