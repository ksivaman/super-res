from PIL import Image
import numpy as np
import glob

for f in glob.glob('dataset/cifar/test/*.png'):
    img = Image.open(f).convert('LA')
    img = img.resize((8, 8))
    img.save(f.replace('cifar', 'cifarsmall'))
