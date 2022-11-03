import torch
import numpy as np
from matplotlib import pyplot as plt

def show(image):
    """ Display image, image should have shape (C, W, H) """
    plt.imshow(np.transpose(image, (1, 2, 0)))
    plt.show()