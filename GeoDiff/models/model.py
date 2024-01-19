import torch
from torch import nn

from encoders import *
from common import *

class DualEncoderEpsNetwork(nn.Module):