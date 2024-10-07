import os
from typing import Literal

import cv2
import numpy as np
import torch
import configargparse

from utils.utils import crop_image, phasemap_8bit, pad_image

homography_res = (1600, 800)


def read_poh(filepath):
    return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)


def color_merge(red, green, blue):
    gbr = cv2.merge([green, blue, red])

    return gbr


def save_poh(test_dir, filename, rgb):
    save_dir = os.path.join(test_dir, 'PoH_4k')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    cv2.imwrite(os.path.join(save_dir, filename), rgb)


def main(poh_dir, filename, mode: Literal["red", "green", "blue", "rgb"]):
    if mode == "rgb":
        red = read_poh(os.path.join(poh_dir, "red", filename))
        green = read_poh(os.path.join(poh_dir, "green", filename))
        blue = read_poh(os.path.join(poh_dir, "blue", filename))

        poh = color_merge(red, green, blue)
    else:
        poh = read_poh(os.path.join(poh_dir, mode, filename))

    poh_4k = cv2.resize(poh, (3840, 2160), interpolation=cv2.INTER_NEAREST)

    save_poh(poh_dir, filename, poh_4k)

if __name__ == "__main__":
    p = configargparse.ArgumentParser()
    p.add_argument('--poh_dir', required=True, type=str, help="Directory of optimized phases (ex) ./phases/{RUN_ID}")
    p.add_argument('--mode', required=True, type=Literal["red", "green", "blue", "rgb"], help="select mode: rgb | green | blue | red")
    p.add_argument('--filename', required=True, type=str, help="target poh image filename (ex) XXX.png")

    opt = p.parse_args()

    main(
        poh_dir=opt.poh_dir,
        mode=opt.mode,
        filename=opt.filename
    )