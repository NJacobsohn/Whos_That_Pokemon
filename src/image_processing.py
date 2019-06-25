from PIL import Image
from collections import defaultdict
import glob
image_dict = defaultdict(list)
for directoryname in glob.glob('../data/dataset/*'):
    image_list = []
    for filename in glob.glob(directoryname):
        im=Image.open(filename)
        image_list.append(im)
    image_dict[directoryname] = image_list