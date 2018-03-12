import numpy as np
import cv2

doPlots = True
doVerbose = True

imgFolder = "images"
imgName0 = "69020.jpg"
imgName1 = "227092.jpg"
imgName2 = "260058.jpg"

rescale = 0.25

# GMMs and EM algorithm implemented as per
# http://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf
class GMM:
    def __init__(self,
                 k, # number of gaussian clusters
                 ):
        self.k = k

    def fit(self,
            img,
            featureType = "I", # can be "I", "RGB", "Lab"
            maxIterations = 20
            ):

        if featureType == "I":
            x = img2IntenstityArray(img)
            dim = 1
        elif featureType == "RGB":
            x = img2rgbArray(img / 255.0)
            dim = 3
        elif featureType == "Lab":
            x = rgb2CieLab(img2rgbArray(img / 255.0))
            dim = 3
        else:
            raise Exception("GMM: Attempting to fit using unsupported feature format.")

        x = x.transpose() # transpose to be N x d
        n, d = x.shape[:2] # number of samples and dimensionality of each sample

        ## Setup GMM parameters

        alphas = np.array([1.0 / self.k for i in range(self.k)], np.float32) # alphas for each Gaussian
        means = x[(np.random.rand(self.k) * n).astype(np.int)] # pick random means for each Gaussian
        covars = np.array([np.eye(d) if d > 1 else [1] for i in range(self.k)], np.float32) # identity covar matrices for each Gaussian

        w = np.zeros((n, self.k), np.float32) # membership weight of each data point to each Gaussian

        logl = [] # log likelihoods over iterations
        iteration = 0
        while iteration < maxIterations:

            print("means", means)
            print("covars", covars)

            # E-step
            p = np.zeros((self.k, n))
            for k in range(self.k):
                xmu = x - means[k]
                print("xmu", xmu)
                det = np.linalg.det(covars[k]) if d > 1 else covars[k]
                inv = np.linalg.inv(covars[k]) if d > 1 else 1.0 / covars[k]

                print("inv", inv)

                exponents = ((np.dot(xmu, inv)) * xmu).sum(axis=1) if d > 1 else (xmu * xmu * inv).sum(axis=1)

                print("EXPONENTS", (exponents < 0).any())

                p[k] = alphas[k] / (((2 * np.pi) ** d * det) ** 0.5) * np.exp(-0.5 * exponents)

            psum = np.sum(p, axis=0)
            for k in range(self.k):
                w[:, k] = p[k] / psum

            # normalize w for each data point (sum of w_k for each point must be 1)
            wmag = np.sum(w, axis=1)
            print("wmag", wmag)
            for k in range(self.k):
                w[:, k] /= wmag

            print("updated w", w)

            # M-step
            N_k = w.sum(axis=0)

            print("nk", N_k)

            alphas = N_k / n # update alpha

            print("alphas", alphas)

            for k in range(self.k):
                means[k] = (1.0 / N_k[k]) * (x * w[:, k].reshape(-1, 1)).sum(axis=0)

                xmmean = x - means[k]
                print("xmmean", xmmean.shape, xmmean.dtype)

                xmeanxt = (xmmean * xmmean).sum(axis=1)
                print("xmeanxt", xmeanxt.shape, xmeanxt.dtype)

                temp = (xmeanxt * w[:, k])
                print("temp", temp.shape)
                temp = temp.sum(axis=0)

                print("temp", temp)
                print(temp.shape, temp.dtype)


                covars[k] = (1.0 / N_k[k]) * temp

            print("updated means", means)
            print("updated covars", covars)



            print("Finished iteration", iteration)
            iteration += 1




# conversion from RGB to Cie L*a*b* color space is done as per
# https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=cvtcolor#cvtcolor
def rgb2CieLab(rgbArray):
    rgb = rgbArray.transpose()

    m = np.array([[0.412453, 0.357580, 0.180423],
                  [0.212671, 0.715160, 0.072169],
                  [0.019334, 0.119193, 0.950227]], np.float32)

    xyz = m.dot(rgb)

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

    lab = np.zeros_like(rgb, np.float32)
    lab[0] = f_L(Y) / 100
    lab[1] = (500.0 * (f_ab(X) - f_ab(Y)) + 128) / 255
    lab[2] = (200.0 * (f_ab(Y) - f_ab(Z)) + 128) / 255

    return lab.astype(np.float32)

def readImg(imgname, imgfolder=imgFolder):
    return cv2.resize(cv2.imread("{}/{}".format(imgfolder, imgname)), (0, 0), fx=rescale, fy=rescale)

def img2rgbArray(img):
    w, h = img.shape[:2]
    ar = img.flatten().reshape((w * h, 3)).transpose()
    return ar.astype(np.float32) / 255

def img2IntenstityArray(img):
    ar = (img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722).reshape((1, -1)).astype(np.float32)
    return ar / 255

def img2LabArray(img):
    return rgb2CieLab(img)

def imshow(imgname, img):
    cv2.imshow(imgname, img.astype(np.uint8))

def main():
    img = readImg(imgName0)
    #
    # a = np.array([[0,0],
    #               [1,2],
    #               [3,2]])
    # b = np.array([[0],
    #               [1],
    #               [2]])
    # print(a)
    # print(b)
    # print(a.shape, b.shape)
    # print(a*b)

    # a = np.array([[0,0],
    #               [1,2],
    #               [3,2]])
    #
    # b = np.array([[0,4],
    #               [1,2]])
    #
    # print()
    #
    # out = np.zeros(a.shape[0])
    # for i in range(a.shape[0]):
    #     out[i] = (a[i].dot(b) * a[i]).sum()
    # print(out)


    gmm = GMM(2)
    gmm.fit(img, "I")

if __name__=="__main__":
    main()