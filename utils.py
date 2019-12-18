'''
Some helper functions.
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init

class AverageMeter(object):
    """Simple class to compute and keep track of a metrics average.
        It could be smarter and more efficient, but it will do.
    """

    def __init__(self, window_size=None):
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.data = []
        self.window_size = window_size

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, weight=1):
        if self.window_size:
            self.data.append(val*weight)
            if self.window_size < len(self.data):
                self.data.pop(0)
            self.sum = sum(self.data)
            self.count = len(self.data)
            self.avg = self.sum / self.count
        else:
            self.sum += val * weight
            self.count += weight
            self.avg = self.sum / self.count
