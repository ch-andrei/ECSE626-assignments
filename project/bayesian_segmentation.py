import numpy as np
import cv2

from project.bouman_orchard_clustering import *

########################################################################################################################

do_verbose = False
do_save_segmented = False
do_show_image = False

########################################################################################################################
# text dumps for silent logging instead of stdout; used when doVerbose is False

logs = []

def dumpLogs(count=50):
    for log in logs[-count:]:
        print(log)

def doprint(*args):
    s = ""
    for arg in args:
        s += str(arg) + " "

    if do_verbose:
        print(s)
        print()
    else:
        logs.append(s)

########################################################################################################################

img_folder = "images"
img_result_folder = "results"
img_name0 = "69020.jpg"
img_name1 = "227092.jpg"
img_name2 = "260058.jpg"
img_name3 = "gandalf.png"

########################################################################################################################

epsilon_precision = 1e-6 # for precision

########################################################################################################################

# configuration
img2read = img_name3
rescale = 1

########################################################################################################################

def readImg(imgname, img_folder=img_folder, rescale=rescale):
    return cv2.resize(cv2.imread("{}/{}".format(img_folder, imgname)), (0, 0), fx=rescale, fy=rescale)

def imshow(imgname, img):
    cv2.imshow(imgname, img.astype(np.uint8))

def img2rgbArray(img):
    w, h = img.shape[:2]
    ar = img.flatten().reshape((w * h, 3)).transpose().astype(np.float32)
    return ar

def readAndPreprocessImage(imgname=img2read):
    img = readImg(imgname)
    x = img2rgbArray(img) / 255
    x = x.transpose() # transpose X to be N x d
    return img, x

def checkNan(val):
    return np.isnan(val) or np.isinf(val) or np.isneginf(val)

########################################################################################################################

def main():
    img, X = readAndPreprocessImage()
    h, w = img.shape[:2]

    tree = ClusterTree(X)
    tree.split()

    img_mask = np.zeros(h * w, np.float)
    cluster_nodes = tree.get_leaf_nodes()
    for index, cluster_node in enumerate(cluster_nodes):
        mask = cluster_node.get_cluster_mask(img)
        img_mask[mask] = index
    img_mask /= img_mask.max()
    img_mask *= 255
    img_mask = img_mask.reshape((h, w))

    imshow("img", img)
    imshow("mask", img_mask)
    cv2.waitKey()


if __name__=="__main__":
    main()
