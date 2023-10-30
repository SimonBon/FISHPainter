import sys

import FISHcreation
from FISHcreation.src.preprocess import get_cell_background
from FISHcreation.src.signals import create_FISH
from FISHcreation.src.process_boxes import merge_boxes_by_color
from CellPatchExtraction import extract_patches

#debug
import matplotlib.pyplot as plt
import os
from pathlib import Path
from cellplot.cellplot.patches import gridPlot, draw_boxes_on_patch

from time import time

for i in range(10):

    s = time()
    background = get_cell_background("/home/simon_g/src/FISHcreation/testdata/IF_RGB.TIFF", normalize=False)[:700, :700]
    print(background.shape)
    print(time()-s)

    s = time()
    patches, masks, _, _, _ = extract_patches(background, "CP_TU", patch_size=128, return_all=True)
    print(len(patches))
    print(time()-s)