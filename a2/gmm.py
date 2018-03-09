import numpy as np
import cv2

doPlots = True
doVerbose = True

imgfolder = "images"
imgname0 = "69020.jpg"
imgname1 = "227092.jpg"
imgname2 = "260058.jpg"

def readGrayImg(imgfolder, imgname):
    return cv2.imread("{}/{}".format(imgfolder, imgname), 0).astype(np.int)

def imshow(imgname, img):
    cv2.imshow(imgname, img.astype(np.uint8))

def main():
    pass

if __name__=="__main__":
    main()