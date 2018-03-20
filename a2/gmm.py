import numpy as np
import cv2
from matplotlib import pyplot as plt

doVerbose = False
dumgLogsOnNan = False
doSaveSegmented = True
doShowImage = False

imgFolder = "images"
imgResultFolder = "results"
imgName0 = "69020.jpg"
imgName1 = "227092.jpg"
imgName2 = "260058.jpg"

# GMM segmentation configuration
img2read = imgName2
rescale = 1
gmmK = 2
gmmMaxIterations = 250 # termination if epsilon criterion isn't reached
gmmFeatureType = "lab"
epsilon = 1e-3 # for convergence
epsilonP = 1e-6 # for precision

# text dumps for silent logging instead of stdout; used when doVerbose is False
logs = []

def dumpLogs(count=50):
    for log in logs[-count:]:
        print(log)

def doprint(*args):
    global logs

    s = ""
    for arg in args:
        s += str(arg) + " "

    if doVerbose:
        print(s)
        print()
    else:
        logs.append(s)

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

def checkNan(val):
    return np.isnan(val) or np.isinf(val) or np.isneginf(val)

def readAndPreprocessImage(imgname=img2read,
                           featureType=gmmFeatureType
                           ):
    img = readImg(imgname)

    featureType = featureType.lower()

    if featureType == "i":
        x = img2IntenstityArray(img) / 255
    elif featureType == "rgb":
        x = img2rgbArray(img) / 255
    elif featureType == "lab":
        x = rgb2CieLab(img2rgbArray(img))
    else:
        raise Exception("Unsupported feature format. Use 'I', 'RGB', or 'LAB'.")

    x = x.transpose() # transpose X to be N x d

    return img, x

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

    lab = np.zeros((2, rgb.shape[1]), np.float32)
    # L = f_L(Y) / 100
    lab[0] = (500.0 * (f_ab(X) - f_ab(Y)) + 127) / 255
    lab[1] = (200.0 * (f_ab(Y) - f_ab(Z)) + 127) / 255

    return lab.astype(np.float32)

# GMMs and EM algorithm implemented as per
# http://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf
class GMM:
    def __init__(self,
                 k, # number of gaussian clusters
                 ):
        self.k = k
        self.trained = False
        self.alphas = None
        self.means = None
        self.covars = None
        self.w = None
        self.loglikelihood = None
        self.bic = None

    def P(self, x, means, covars, d, k):
        xmu = x - means[k]
        det = np.linalg.det(covars[k])
        inv = np.linalg.inv(covars[k])

        exponents = ((np.dot(xmu, inv)) * xmu).sum(axis=1) if d > 1 else (xmu * xmu * inv).sum(axis=1)

        return 1.0 / ((2 * np.pi) ** d * det) ** 0.5 * np.exp(-0.5 * exponents)

    # predict only
    def predict(self, x):
        if not self.trained:
            raise Exception("GMM: Attempting to predict using untrained GMM.")

        n, d = x.shape[:2] # number of samples and dimensionality of each sample

        # rescale to full range
        x -= x.min() - epsilonP # epsilon to avoid 0 and 1 values
        x /= x.max() + epsilonP # epsilon to avoid 0 and 1 values

        w = np.zeros((n, self.k), np.float32) # membership weight of each data point to each Gaussian

        # compute raw probabilities
        p = np.zeros((self.k, n))
        for k in range(self.k):
            p[k] = self.P(x, self.means, self.covars, d, k)

        # multiply probabilities by alpha values
        p *= self.alphas.reshape(self.k, 1)

        psum = p.sum(axis=0)

        # compute BIC value
        bic = -2.0 * np.log(psum).sum() + np.log(n) * self.bic_k

        # compute log likelihood value
        loglikelihood = np.log10(psum).sum()

        # compute membership weights
        for k in range(self.k):
            w[:, k] = p[k] / (psum + epsilonP)  # add epsilon to avoid division by zero

        return w, loglikelihood, bic

    def fit(self,
            x,
            gmmMaxIterations=gmmMaxIterations,
            epsilon=epsilon
            ):
        n, d = x.shape[:2] # number of samples and dimensionality of each sample

        # rescale to full range
        x -= x.min() - epsilonP # add epsilon to avoid 0 and 1 values
        x /= x.max() + epsilonP # add epsilon to avoid 0 and 1 values

        alphas = np.array([1.0 / self.k for _ in range(self.k)], np.float) # alphas for each Gaussian
        covars = np.array([np.eye(d) if d > 1 else [[1]] for _ in range(self.k)], np.float) # identity covar matrices for each Gaussian
        means = x[(np.random.rand(self.k) * n).astype(np.int)] # pick random means for each Gaussian

        w = np.zeros((n, self.k), np.float32) # membership weight of each data point to each Gaussian

        # constant for computing BIC
        # k model complexity: number of means params (k*d) + number of covar params (k*d*d) + number of alpha params (k)
        self.bic_k = self.k * d + self.k * d * d + self.k

        loglikelihoods = [] # stores log likelihoods over iterations
        bics = [] # stores BIC values
        iteration = 0
        while iteration < gmmMaxIterations:

            ########## E-step ##########

            # compute raw probabilities
            p = np.zeros((self.k, n))
            for k in range(self.k):
                p[k] = self.P(x, means, covars, d, k)

            # multiply probabilities by alpha values
            p *= alphas.reshape(self.k, 1)

            # now update BIC and log likelihoods: this is an optimization -> for speedup
            # its faster to do it here than after updating alphas, means, covars, because we don't need to recompute
            # the probabilities for each Gaussian for each X.
            # this essentially delays convergence check by 1 iteration, since we compute likelihood before updating.
            # this does not affect correctness, only the timing of the check

            psum = p.sum(axis=0)

            # update BIC values
            bic = -2.0 * np.log(psum).sum() + np.log(n) * self.bic_k
            bics.append(bic)

            # update log likelihood values
            loglikelihood = np.log10(psum).sum()
            loglikelihoods.append(loglikelihood)

            print("\rRunning iteration {}; current loglike = [{}], bic [{}]".format(iteration, loglikelihood, bic), end="")

            # compute membership weights
            for k in range(self.k):
                w[:, k] = p[k] / (psum + epsilonP) # add epsilon to avoid division by zero

            ########## M-step ##########

            N_k = w.sum(axis=0)

            # update alphas for each Gaussian
            alphas = N_k / w.sum()

            for k in range(self.k):
                # update mean
                means[k] = (1.0 / N_k[k]) * (x * w[:, k].reshape(-1, 1)).sum(axis=0)

                # now we must compute dot products of MANY PAIRS of vectors INDEPENDENTLY
                # I could not figure out a good vectorized way to do this using built in functions so I improvised
                # example:
                # for d=3 and n=2, X is [a b c, d e f] which is 2 vectors of size 3
                # we want individual dot products [a b c]T dot [a b c], [d e f]T dot [d e f], so two 3x3 matrices
                # individually, for each of the N 1d vectors f form [a b c], we would get the dot product
                #                        [aa ab ac]
                # [a b c]T dot [a b c] = [ab bb bc]
                #                        [ac bc cc]
                # for N 1d vectors, we can do this by creating the following matrices and simple element multiplication
                # [a a a] [a b c]   [aa ab ac]
                # [b b b] [a b c]   [ab bb bc]
                # [c c c] [a b c] = [ac bc cc]
                # [d d d] [d e f] = [dd de df]
                # [e e e] [d e f]   [de ee ef]
                # [f f f] [d e f]   [df de df]
                wk = w[:, k].reshape(n, 1, 1)
                xmmean = x - means[k]
                r1 = np.repeat(xmmean, d)
                r2 = np.repeat(xmmean, d, axis=0).flatten()
                r3 = (r1 * r2).reshape(n, d, d)
                temp = (r3 * wk).sum(0)

                covars[k] = (1.0 / N_k[k]) * temp

            doprint()
            doprint("alphas updated", alphas)
            doprint("means updated", means)
            doprint("covars updated", covars)

            if checkNan(loglikelihood):
                if not doVerbose and dumgLogsOnNan:
                    dumpLogs()
                break

            iteration += 1
            if iteration > 1 and np.abs(loglikelihood - loglikelihoods[-2]) < epsilon:
                # stop if convergence has been achieved
                break

        # store the final values
        self.alphas = alphas
        self.means = means
        self.covars = covars
        self.w = w
        self.loglikelihood = loglikelihoods[-1]
        self.bic = bics[-1]
        self.trained = True

def doKFoldCrossValidation(imgname=img2read,
                           k_crossValidation=5,
                           featureType=gmmFeatureType,
                           maxK=8
                           ):
    img, x = readAndPreprocessImage(imgname=imgname, featureType=featureType)

    ks = []
    logkey_train = []
    logkey_test = []
    bickey_train = []
    bickey_test = []

    for k_gmm in range(2, maxK+1):
        xshuf = x.copy()
        np.random.shuffle(xshuf) # randomize ordering of the vector X

        n, d = x.shape[:2]  # number of samples and dimensionality of each sample

        # this won't use every single data point, but that's fine
        n_crossval = int(n / k_crossValidation)

        gmm = GMM(k_gmm)

        logl_train_avg = 0
        logl_test_avg = 0
        bic_train_avg = 0
        bic_test_avg = 0

        for k_crossVal in range(k_crossValidation):
            print("\nRunning crossval with {} Gaussians on set {}".format(k_gmm, k_crossVal))

            # mask to separate validation and training sets
            mask = np.zeros(n, np.bool)
            mask[n_crossval*k_crossVal: n_crossval*(k_crossVal+1)] = 1

            xt = xshuf[mask == 0] # training
            yt = xshuf[mask == 1] # testing

            # fit on train data
            gmm.fit(xt)
            logl_train_avg +=  gmm.loglikelihood
            bic_train_avg += gmm.bic

            # predict on test data
            _, logl_test, bic_test = gmm.predict(yt)
            logl_test_avg += logl_test
            bic_test_avg += bic_test

        # average cross validation results
        logl_train_avg /= k_crossValidation
        logl_test_avg /= k_crossValidation
        bic_train_avg /= k_crossValidation
        bic_test_avg /= k_crossValidation

        # put into lists for plotting
        ks.append(k_gmm)
        logkey_train.append(logl_train_avg)
        logkey_test.append(logl_test_avg)
        bickey_train.append(-bic_train_avg)
        bickey_test.append(-bic_test_avg)

    # plot the results

    # PLOT TRAIN
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Number of Gaussians')
    ax1.set_ylabel("Log Likelihood", color=color)
    ax1.plot(ks, logkey_train, 'o-', color=color)

    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel("Bayesian Information Criterion", color=color)
    ax2.plot(ks, bickey_train, 'o-', color=color)

    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()

    # PLOT TEST
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Number of Gaussians')
    ax1.set_ylabel("Log Likelihood", color=color)
    ax1.plot(ks, logkey_test, 'o-', color=color)

    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel("Bayesian Information Criterion", color=color)
    ax2.plot(ks, bickey_test, 'o-', color=color)

    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()


def doGMMsegmentation(imgname=img2read,
                      k=gmmK,
                      featureType=gmmFeatureType,
                      saveSegmented=doSaveSegmented,
                      showImage=doShowImage
                      ):
    img, x = readAndPreprocessImage(imgname=imgname, featureType=featureType)

    h, w = img.shape[:2]

    if k < 2:
        raise Exception("GMM: selected K that is too small.")

    print("Running '{}' features GMM segmentation on image {} with {} Gaussians...".format(featureType, imgname, k))

    gmm = GMM(k)
    gmm.fit(x)

    colors = 255 * np.random.rand(k, 3) # random colors to display GMM segmentation results
    labels = np.argmax(gmm.w, axis=1).reshape((h, w))
    u, c = np.unique(labels, return_counts=True)
    seg = np.zeros_like(img)
    for i in range(k):
        seg[labels == i] = colors[i]

    print("Results: ", u, c)

    if showImage:
        cv2.imshow("img", img)
        cv2.imshow("GMM-{}".format(k), seg)
        cv2.waitKey()

    if saveSegmented:
        imgname2write ="{}/{}-segmented-gmm{}-{}.png".format(imgResultFolder, imgname.split('.')[0], k, gmmFeatureType)
        cv2.imwrite(imgname2write, seg)

def main():

    # doKFoldCrossValidation()

    #####################

    # do segmentation and save images
    for k in range(2, 7, 2):
        doGMMsegmentation(k=k)

if __name__=="__main__":
    main()