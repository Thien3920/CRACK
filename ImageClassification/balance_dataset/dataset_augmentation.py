
from ToolMakeNoise.make_noise.tool import rain, sun, smoke,rotation
from glob import glob
import argparse
import random
import cv2
import os



parser = argparse.ArgumentParser()
parser.add_argument('--input_dir',default = './Data/error_train', help='input directory path')
parser.add_argument('--output_dir', default = './Data/train_augmentation_dataset',help='output directory path')
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

        func = ['rain', 'smoke','sun']
        ###
        for angle in [90 , -90 , 180]:
            img = rotation(I,angle)

            f = random.choice(func)
            if f == "rain":
                out = rain(img)
            elif f == "smoke":
                out = smoke(img)
            else:
                 out = sun(img)

            # out = K = cv2.hconcat([I, out])

            dst = os.path.join(output_dir,i,f+"_"+str(angle)+"_"+image_name)
            cv2.imwrite(dst,out)
