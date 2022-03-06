from utils.image_processor import *
from glob import glob
import cv2
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir',default = './Data/error_train', help='input directory path')
parser.add_argument('--output_dir', default = './Data/argument_dataset',help='output directory path')
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir


if not os.path.exists(output_dir):
    for i in ['', 'cracked', 'uncracked']:
        os.mkdir(os.path.join(output_dir, i))

for i in ["cracked","uncracked"]:
    path = os.path.join(input_dir,i)

    list_images = glob(path + '/*.jpg') + glob(path + '/*.jpeg') + glob(path + '/*.png')
    list_images.sort()

    for J in list_images:
        image_name = J.split("/")[-1]
        I = cv2.imread(J)

        for angle in [90 , -90 , 180]:
            img = rotate_image(I,angle)
            img = blur_image(img, dsize=5)

            dst = os.path.join(output_dir,i,str(angle)+"_"+image_name)
            cv2.imwrite(dst,img)
