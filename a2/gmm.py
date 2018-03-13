import numpy as np
import cv2

doPlots = True

doVerbose = False
# doVerbose = True

imgFolder = "images"
imgName0 = "69020.jpg"
imgName1 = "227092.jpg"
imgName2 = "260058.jpg"

rescale = 1
gmmK = 2
gmmMaxIterations = 500
gmmFeatureType = "lab"
img2read = imgName0

# GMMs and EM algorithm implemented as per
# http://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf
class GMM:
    def __init__(self,
                 k, # number of gaussian clusters
                 ):
        self.k = k

    def P(self, x, means, covars, d, k):
        xmu = x - means[k]

        det = np.linalg.det(covars[k])
        inv = np.linalg.inv(covars[k])

        exponents = ((np.dot(xmu, inv)) * xmu).sum(axis=1) if d > 1 else (xmu * xmu * inv).sum(axis=1)

        return 1.0 / ((2 * np.pi) ** d * det) ** 0.5 * np.exp(-0.5 * exponents)

    def fit(self,
            img,
            featureType = "i", # can be "I", "RGB", "Lab"
            gmmMaxIterations = gmmMaxIterations,
            epsilon=1e-4
            ):

        featureType = featureType.lower()

        if featureType == "i":
            x = img2IntenstityArray(img) / 255
        elif featureType == "rgb":
            x = img2rgbArray(img) / 255
        elif featureType == "lab":
            x = rgb2CieLab(img2rgbArray(img))
        else:
            raise Exception("GMM: Attempting to fit using unsupported feature format. Use 'I', 'RGB', or 'LAB'.")

        x = x.transpose() # transpose to be N x d
        n, d = x.shape[:2] # number of samples and dimensionality of each sample

        alphas = np.array([1.0 / self.k for i in range(self.k)], np.float32) # alphas for each Gaussian
        means = x[(np.random.rand(self.k) * n).astype(np.int)] # pick random means for each Gaussian
        covars = np.array([np.eye(d) if d > 1 else [[1]] for i in range(self.k)], np.float32) # identity covar matrices for each Gaussian

        w = np.zeros((n, self.k), np.float32) # membership weight of each data point to each Gaussian

        logl = [1000000000000] # log likelihoods over iterations
        iteration = 0
        while iteration < gmmMaxIterations:

            # E-step
            p = np.zeros((self.k, n))
            for k in range(self.k):
                p[k] = alphas[k] * self.P(x, means, covars, d, k)

            psum = np.sum(p, axis=0)

            for k in range(self.k):
                w[:, k] = p[k] / psum

            # normalize w for each data point (sum of w_k for each point must be 1)
            wmag = np.sum(w, axis=1)

            for k in range(self.k):
                w[:, k] /= wmag

            # M-step
            N_k = w.sum(axis=0)
            N_ksum = w.sum()

            alphas = N_k / N_ksum # update alpha

            for k in range(self.k):
                means[k] = (1.0 / N_k[k]) * (x * w[:, k].reshape(-1, 1)).sum(axis=0)

                wk = w[:, k].reshape(n, 1, 1)

                xmmean = x - means[k]

                r1 = np.repeat(xmmean, d)
                r2 = np.repeat(xmmean, d, axis=0).flatten()
                r3 = (r1 * r2).reshape(n, d, d)
                temp = (r3 * wk).sum(0)

                covars[k] = (1.0 / N_k[k]) * temp

            # compute loglikelihood
            p = np.zeros((self.k, n))
            for k in range(self.k):
                p[k] = alphas[k] * self.P(x, means, covars, d, k)

            loglike = np.log10(p.sum(axis=0)).sum()
            logl.append(loglike)

            print("\rFinished iteration {} with log loglikelihood [{}]".format(iteration, loglike), end="")
            iteration += 1

            if iteration > 1 and np.abs(loglike - logl[-2]) < epsilon:
                break

        self.w = w

def doprint(*args):
    if doVerbose:
        for arg in args:
            print(arg, end=" ")
        print("")

# conversion from RGB to Cie L*a*b* color space is done as per
# https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=cvtcolor#cvtcolor
def rgb2CieLab(rgbArray):
    # input rgb as float 0-1 and a 3 x N array
    rgb = rgbArray / 255

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
    ar = img.flatten().reshape((w * h, 3)).transpose().astype(np.float32)
    return ar

def img2IntenstityArray(img):
    ar = (img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722).reshape((1, -1)).astype(np.float32)
    return ar

def img2LabArray(img):
    return rgb2CieLab(img)

def imshow(imgname, img):
    cv2.imshow(imgname, img.astype(np.uint8))

def main():
    # a = np.array([[0,0],
    #               [1,2],
    #               [3,2]])
    #
    # b = np.array([[0,4],
    #               [1,2]])
    #
    # doprint()
    #
    # out = np.zeros(a.shape[0])
    # for i in range(a.shape[0]):
    #     out[i] = (a[i].dot(b) * a[i]).sum()
    # doprint(out)

#####################

    img = readImg(img2read)
    w, h = img.shape[:2]

    k = gmmK
    if k < 2:
        raise Exception("GMM: selected K that is too small.")

    gmm = GMM(k)
    gmm.fit(img, gmmFeatureType)

    colors = 255 * np.random.rand(k, 3) # random colors to display GMM segmentation results
    labels = np.argmax(gmm.w, axis=1).reshape((w, h))
    u, c = np.unique(labels, return_counts=True)
    seg = np.zeros_like(img)
    for i in range(k):
        seg[labels == i] = colors[i]

    print("Results: ", u, c)

    cv2.imshow("img", img)
    cv2.imshow("GMM-{}".format(k), seg)
    cv2.waitKey()

# #####################
#     d = 3
#     wk = np.array([0.25,0.25,0.5,0.3,0.1]).reshape(5, 1,1)
#     count = wk.shape[0]
#     xmean = np.arange(d*count).reshape(count, d)
#
#     r1 = np.repeat(xmean,d)
#     r2 = np.repeat(xmean,d,axis=0).flatten()
#     r3 = (r1 * r2).reshape(count, d, d)
#     r4 = (r3 * wk).sum(0) / count
#
#     c = np.array([[0, 1, 2]])
#     c1 = c.transpose().dot(c) * wk[0]
#     c = np.array([[3, 4, 5]])
#     c2 = c.transpose().dot(c) * wk[1]
#     c = np.array([[6, 7, 8]])
#     c3 = c.transpose().dot(c) * wk[2]
#     c = np.array([[9, 10, 11]])
#     c4 = c.transpose().dot(c) * wk[3]
#     c = np.array([[12, 13, 14]])
#     c5 = c.transpose().dot(c) * wk[4]
#     c3 = (c1 + c2 + c3 + c4 + c5) / 5
#
#     print(r4)
#     print(c3)




if __name__=="__main__":
    main()