import numpy as np
import cv2

doPlots = True
doVerbose = True

imgFolder = "images"
imgName0 = "69020.jpg"
imgName1 = "227092.jpg"
imgName2 = "260058.jpg"

# conversion from RGB to Cie L*a*b* color space is done as per
# https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=cvtcolor#cvtcolor
def rgb2CieLab(rgb):

    print(rgb)

    m = np.array([[0.412453, 0.357580, 0.180423],
                  [0.212671, 0.715160, 0.072169],
                  [0.019334, 0.119193, 0.950227]], np.float)

    xyz = m.dot(rgb)

    print(xyz)

    X, Y, Z = (xyz[0], xyz[1], xyz[2])
    X /= 0.950456
    Z /= 1.088754

    def f_L(t, thresh=0.008856):
        L = np.zeros_like(t)
        inds = t > thresh
        ninds = (np.logical_not(inds))
        if inds.any():
            L[inds] = 116.0 * np.power(t[inds], 1.0 / 3.0) - 16.0
        if ninds.any():
            L[ninds] = 903.3 * t[ninds]
        return L

    def f_ab(t, thresh=0.008856):
        ab = np.zeros_like(t)
        inds = t > thresh
        ninds = np.logical_not(inds)
        if inds.any():
            ab[inds] = np.power(t[inds], 1.0 / 3.0)
        if ninds.any():
            ab[ninds] = 7.787 * t[ninds] + 16.0 / 116.0
        return ab

    lab = np.zeros_like(rgb, np.float)
    lab[0] = f_L(Y)
    lab[1] = 500.0 * (f_ab(X) - f_ab(Y))
    lab[2] = 200.0 * (f_ab(Y) - f_ab(Z))

    return lab

def readImg(imgname, imgfolder=imgFolder):
    return cv2.imread("{}/{}".format(imgfolder, imgname))

def img2rgbArray(img):
    w, h = img.shape[:2]
    ar = img.flatten().reshape((w * h, 3)).transpose()
    return ar

def img2IntenstityArray(img):
    return (img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722).reshape((1, -1))

def img2LabArray(img):
    return rgb2CieLab(img)

def imshow(imgname, img):
    cv2.imshow(imgname, img.astype(np.uint8))

def main():
    img = readImg(imgName0)

    img = np.array(
        [
            [
                [1,2,3],[4,5,6],[7,8,9],[10,11,12]
            ]
        ]
    )

    print(img.shape)

    imgArray = img2rgbArray(img)

    lab = img2LabArray(imgArray)

if __name__=="__main__":
    main()