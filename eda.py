#!/usr/bin/python

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
import argparse

### PATHS ###
TRAIN_DIR = './datasets/data/traindev/'
TEST_DIR = './datasets/data/test/'


### ARGUMENTS ###
def get_args():
    args = argparse.ArgumentParser(description='Image Forgery Detection Dataset EDA')
    args.add_argument('--dims', action='store_true', help='Create a scatter plot of image dimensions')

    return args.parse_args()

### EDA ###

# Util: Get image dimensions
def get_dims(file):
    img = cv2.imread(file)
    h,w = img.shape[:2]
    return h,w


# EDA: Scatter plot of image dimensions
def create_dimensions_scatter(filelist):
    dims = {'height': [], 'width': []}
    for file in tqdm(filelist):
        dims['height'].append(get_dims(file)[0])
        dims['width'].append(get_dims(file)[1])

    dim_df = pd.DataFrame(dims, columns=['height', 'width'])
    # print(dim_df.head())
    sizes = dim_df.groupby(['height', 'width']).size().reset_index().rename(columns={0:'count'})
    print(sizes)

    colors = np.random.rand(len(sizes['count']))
    fig = plt.figure(figsize=(8,6))
    plt.scatter(x=sizes['height'], y=sizes['width'], s=sizes['count']*8, c=colors, alpha=0.75)
    plt.xlabel('Height')
    plt.ylabel('Width')
    plt.title('Image Dimensions')
    plt.grid(True)
    plt.savefig('eda_results/eda_dim_img.png')


if __name__ == '__main__':
    args = get_args()

    test_authentic_path = TEST_DIR + "authentic/"
    test_authentic = [test_authentic_path + f for f in os.listdir(test_authentic_path)]
    if args.dims:
        print(f"[+] Creating image dimensions scatter plot for test/authentic ({len(test_authentic)}) images.")
        create_dimensions_scatter(test_authentic)