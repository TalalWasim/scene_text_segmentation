import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

img_folder = './ICDAR_18/model_best_epochs-100_LR-0001_WD-0001/'
res_folder = os.path.join(img_folder,'matplot_vis')

os.makedirs(res_folder, exist_ok=True)

files = glob.glob(img_folder + '*.png')

for i in files:
    img = cv2.imread(i)

    img_int = np.zeros(img.shape)
    img_int[img==1] = 255
    img_int = img_int.astype(int)

    plt.figure()
    plt.imshow(img_int)

    fname = i.split('\\')[-1]
    path = os.path.join(res_folder,fname)
    
    plt.savefig(path)
    plt.close()
