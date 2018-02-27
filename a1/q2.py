import numpy as np
import cv2

import plotly.plotly as py
import plotly.graph_objs as go

doPlots = True
doVerbose = True

imgfolder = "images"
imgname000 = "000.png"
imgname001 = "001.png"
imgname002 = "002.png"
imgname003 = "003.png"
imgname004 = "004.png"
imgname005 = "005.png"

def readGrayImg(imgfolder, imgname):
    return cv2.imread("{}/{}".format(imgfolder, imgname), 0).astype(np.int)

def imshow(imgname, img):
    cv2.imshow(imgname, img.astype(np.uint8))

def randomNoiseImg(n, m, maxval):
    return (np.random.rand(n, m) * 2 * maxval - maxval).astype(np.int)

# single image
def computeProbabilities(img):
    uniques, counts = np.unique(img, return_counts=True)
    p = counts / counts.sum()
    return p

# between two images
# returns 3 lists: ordered P(x,y) and corresponding P(x) and P(y) in order of appearance in P(x,y) (with duplicates)
# ex:
#     P(x,y) = [P(122,15), P(76, 15)]
#     P(x)   = [P(122)   , P(76)    ]
#     P(y)   = [P(15)    , P(15)    ]
# this format allows for vectorized computation later
def computeProbabilities2(img1, img2):
    img12 = np.zeros((2,) + img1.shape[:2], np.int)
    img12[0] = img1
    img12[1] = img2
    img12 = img12.reshape((2, -1))

    uniques1, counts1 = np.unique(img1, return_counts=True)
    uniques2, counts2 = np.unique(img2, return_counts=True)
    uniques12, counts12 = np.unique(img12, return_counts=True, axis=1)

    p1 = counts1 / counts1.sum()
    p2 = counts2 / counts2.sum()
    p12 = counts12 / counts12.sum()

    d1 = dict(zip(uniques1, p1))
    d2 = dict(zip(uniques2, p2))

    p1 = []
    p2 = []
    for i in range(uniques12.shape[1]):
        p1.append(d1[uniques12[0, i]])
        p2.append(d2[uniques12[1, i]])

    p1 = np.array(p1)
    p2 = np.array(p2)

    return p1, p2, p12

def computeImgEntropy(a):
    p = computeProbabilities(a)
    return -np.sum(p * np.log2(p))

def computeJointEntropy(a, b):
    p1, p2, p12 = computeProbabilities2(a, b)
    return -np.sum(p12 * np.log2(p12))

def computeMutualInformation(a, b):
    p1, p2, p12 = computeProbabilities2(a, b)
    return np.sum(p12 * np.log2(p12 / p1 / p2))

def computeKullbackLeiblerDivergence(a, b):
    p1, p2, p12 = computeProbabilities2(a, b)
    return np.sum(p1 * np.log(p1 / p2))

def computeMse(img1, img2):
    dif = img1 - img2
    return (dif * dif).sum()

def combineImgs(img1, img2, clamp=False):
    imgCombined = img1 + img2
    return np.clip(imgCombined, 0, 255) if clamp else imgCombined

def q2a(verbose=doVerbose, imgname=imgname000):
    img = readGrayImg(imgfolder, imgname)

    entropy = computeImgEntropy(img)

    if verbose:
        print("a) The entropy of the image {} is [{}].".format(imgname, entropy))

def q2b(verbose=doVerbose, imgname=imgname000):
    img = readGrayImg(imgfolder, imgname)

    noiseAmplitude = 20
    img = randomNoiseImg(img.shape[0], img.shape[1], noiseAmplitude)
    entropy = computeImgEntropy(img)

    if verbose:
        print("b) The entropy of a random noise image with noise amplitude {} is [{}].".format(noiseAmplitude, entropy))

def q2c(noiseAmplitude=20, verbose=doVerbose, imgname=imgname000):
    img = readGrayImg(imgfolder, imgname)
    entropy = computeImgEntropy(img)

    imgNoise = randomNoiseImg(img.shape[0], img.shape[1], noiseAmplitude)
    entropyNoisy = computeImgEntropy(imgNoise)

    imgCombined = combineImgs(img, imgNoise)
    entropyCombined = computeImgEntropy(imgCombined)

    if verbose:
        print("c) Entropy of image   : [{}].".format(entropy))
        print("   Entropy of noise   : [{}].".format(entropyNoisy))
        print("   Entropy of combined: [{}].".format(entropyCombined))

    return noiseAmplitude, entropy, entropyNoisy, entropyCombined

def q2d(plot=doPlots):
    print("d) plotting...")

    a = []
    e_i = []
    e_n = []
    e_c = []

    for noiseAmplitude in range(0, 201, 5):
        results = q2c(noiseAmplitude, verbose=False)
        a.append(results[0])
        e_i.append(results[1])
        e_n.append(results[2])
        e_c.append(results[3])

    if plot:
        # Create traces
        trace0 = go.Scatter(
            x=a,
            y=e_i,
            mode='lines+markers',
            name='Image'
        )
        trace1 = go.Scatter(
            x=a,
            y=e_n,
            mode='lines+markers',
            name='Noise'
        )
        trace2 = go.Scatter(
            x=a,
            y=e_c,
            mode='lines+markers',
            name='Combined'
        )
        data = [trace0, trace1, trace2]

        # Edit the layout
        layout = dict(title='Entropy as a function of noise amplitude',
                      xaxis=dict(title='Noise Amplitude'),
                      yaxis=dict(title='Entropy'),
                      )

        fig = dict(data=data, layout=layout)

        py.plot(fig, filename='line-mode')
    else:
        print("Skipped.")

def q22ab(verbose=doVerbose):
    img = readGrayImg(imgfolder, imgname001)

    noiseAmplitude = 20
    imgNoise = randomNoiseImg(img.shape[0], img.shape[1], noiseAmplitude)

    en1 = computeImgEntropy(img)
    en2 = computeImgEntropy(imgNoise)
    jen = computeJointEntropy(img, imgNoise)
    mi = computeMutualInformation(img, imgNoise)

    if verbose:
        print("ab) the MI of the image and 20-noise is [{}]\n"
              "   entropy: img [{}], noise [{}].\n"
              "   joint e: [{}]".format(mi, en1, en2, jen))

def q22c(noiseAmplitude = 20, verbose=doVerbose):
    img = readGrayImg(imgfolder, imgname001)

    imgNoise1 = randomNoiseImg(img.shape[0], img.shape[1], noiseAmplitude)
    imgNoise2 = randomNoiseImg(img.shape[0], img.shape[1], noiseAmplitude)

    imgCombined = combineImgs(img, imgNoise1)

    kl_n2n = computeKullbackLeiblerDivergence(imgNoise1, imgNoise2)

    kl_i2n = computeKullbackLeiblerDivergence(img, imgNoise1)
    kl_i2ni = computeKullbackLeiblerDivergence(img, imgCombined)
    mi_i2ni = computeMutualInformation(img, imgCombined)

    if verbose:
        print("c) Kullback-Leibler divergence:\n"
              "   noise to noise : [{}]\n"
              "   img to noise   : [{}]\n"
              "   img to noisyimg: [{}]\n"
              "   mi img to noisy: [{}]".format(kl_n2n, kl_i2n, kl_i2ni, mi_i2ni))

    return {"a": noiseAmplitude, "mi": mi_i2ni, "kl": kl_i2ni}

def q22de(plot=doPlots):
    print("d) plotting...")

    a = []
    mi = []
    kl = []

    for noiseAmplitude in range(0, 201, 5):
        results = q22c(noiseAmplitude, verbose=False)
        a.append(results["a"])
        mi.append(results["mi"])
        kl.append(results["kl"])

    if plot:
        # plot MI
        trace0 = go.Scatter(
            x=a,
            y=mi,
            mode='lines+markers',
            name='mi'
        )
        data = [trace0]

        # Edit the layout
        layout = dict(title='Mutual Information as a function of noise amplitude',
                      xaxis=dict(title='Noise Amplitude'),
                      yaxis=dict(title='Mutual Information'),
                      )

        fig = dict(data=data, layout=layout)
        py.plot(fig, filename='mi')

        # plot KL
        trace1 = go.Scatter(
            x=a,
            y=kl,
            mode='lines+markers',
            name='kl'
        )
        data = [trace1]

        # Edit the layout
        layout = dict(title='Kullback-Leibler divergence as a function of noise amplitude',
                      xaxis=dict(title='Noise Amplitude'),
                      yaxis=dict(title='KL divergence'),
                      )

        fig = dict(data=data, layout=layout)
        py.plot(fig, filename='kldiv')
    else:
        print("Skipped.")

from scipy.ndimage.interpolation import shift

def q23(verbose=doVerbose, imgname1=imgname000, imgname2=imgname001):
    img1 = readGrayImg(imgfolder, imgname1)
    img2 = readGrayImg(imgfolder, imgname2)

    rows, cols = img1.shape

    bestMi = 0
    bestMse = 255*255*rows*cols

    for i in range(81):
        for j in range(81):
            x = i - 40
            y = j - 40

            # translate
            imgTranslated = shift(img1, (x, y))

            mi = computeMutualInformation(imgTranslated, img2)
            mse = computeMse(imgTranslated, img2)

            if mse < bestMse:
                bestMse = mse
                bestMseXy = (x, y)

            if mi > bestMi:
                bestMi = mi
                bestMiXy = (x, y)

        print("\rfinished i {}".format(i), end="")

    if verbose:
        print("\n"
              "   best mse [{}]: xy [{}]\n"
              "   best mi  [{}]: xy [{}]".format(bestMse, bestMseXy, bestMi, bestMiXy))

def q22showshift(verbose=doVerbose):
    """
    results from running the searches
    a) best mse [0]: xy [(15, 12)]
       best mi  [3.221302964559462]: xy [(15, 12)]
    b) best mse [185532018]: xy [(9, -8)]
       best mi  [1.2698222716441556]: xy [(10, 3)]
    c) best mse [150624940]: xy [(0, 19)]
       best mi  [0.6098016028317651]: xy [(1, 31)]
    """
    results = {
        (imgname000, imgname001): [(15,12), (15,12)],
        (imgname002, imgname003): [(9,-8),  (10,3 )],
        (imgname004, imgname005): [(0,19),  (1, 31)]
    }

    for key in results:
        imgname1, imgname2 = key

        img1 = readGrayImg(imgfolder, imgname1)
        img2 = readGrayImg(imgfolder, imgname2)

        imshow(imgname1, img1)
        imshow(imgname2, img2)

        shifts = results[key]
        for index, (x, y) in enumerate(shifts):
            imgTranslated = shift(img1, (x, y))

            name = "mse blend" if index == 0 else "mi blend"

            imgBlend = (img2.astype(np.float) + imgTranslated.astype(np.float)) / 2

            imshow(name, imgBlend)

        cv2.waitKey()


def main():
    print("\n2.1: Entropy")
    q2a()
    q2b()
    q2c()
    q2d()

    print("\n2.2 MI and KL divergence")
    q22ab()
    q22c()
    q22de()

    # print("\n2.3 Simple Image Registration")
    # q23(imgname1=imgname000, imgname2=imgname001)
    # q23(imgname1=imgname002, imgname2=imgname003)
    # q23(imgname1=imgname004, imgname2=imgname005)

    # q22showshift()

if __name__=="__main__":
    main()