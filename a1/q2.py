import numpy as np
import cv2

import plotly.plotly as py
import plotly.graph_objs as go

imgfolder = "images"
imgname = "001.png"

def readGrayImg(imgfolder, imgname):
    return cv2.imread("{}/{}".format(imgfolder, imgname), 0).astype(np.int)

def randomNoiseImg(n, m, maxval):
    return (np.random.rand(n, m) * 2 * maxval - maxval).astype(np.int)

def computeProbabilities(img):
    uniques, counts = np.unique(img, return_counts=True)
    p = counts / counts.sum()
    return p

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

    d1 = dict(zip(uniques1, p1))
    d2 = dict(zip(uniques2, p2))

    p1 = []
    p2 = []
    for i in range(uniques12.shape[1]):
        v1 = uniques12[0, i]
        v2 = uniques12[1, i]

        p1.append(d1[v1])
        p2.append(d2[v2])

    p1 = np.array(p1)
    p2 = np.array(p2)
    p12 = counts12 / counts12.sum()

    return p1, p2, p12

def computeImgEntropy(img):
    p = computeProbabilities(img)
    return -np.sum(p * np.log2(p))

def computeMutualInformation(img1, img2):
    p1, p2, p12 = computeProbabilities2(img1, img2)
    return -np.sum(p12 * np.log2(p12 / p1 / p2))

def q2a(verbose=True):
    img = readGrayImg(imgfolder, imgname)

    entropy = computeImgEntropy(img)

    if verbose:
        print("a) The entropy of the image {} is [{}].".format(imgname, entropy))

def q2b(verbose=True):
    img = readGrayImg(imgfolder, imgname)

    noiseAmplitude = 20
    img = randomNoiseImg(img.shape[0], img.shape[1], noiseAmplitude)
    entropy = computeImgEntropy(img)

    if verbose:
        print("b) The entropy of a random noise image with noise amplitude {} is [{}].".format(noiseAmplitude, entropy))

def q2c(noiseAmplitude=20, verbose=True):
    img = readGrayImg(imgfolder, imgname)
    entropy = computeImgEntropy(img)

    imgNoise = randomNoiseImg(img.shape[0], img.shape[1], noiseAmplitude)
    entropyNoisy = computeImgEntropy(imgNoise)

    imgCombined = np.clip(img + imgNoise, 0, 255)
    entropyCombined = computeImgEntropy(imgCombined)

    if verbose:
        print("c) Entropy of image   : [{}].".format(entropy))
        print("   Entropy of noise   : [{}].".format(entropyNoisy))
        print("   Entropy of combined: [{}].".format(entropyCombined))

    return noiseAmplitude, entropy, entropyNoisy, entropyCombined

def q2d(plot=False):
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

from scipy.stats import chi2_contingency

def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")
    mi = 0.5 * g / c_xy.sum()
    return mi

def q22a(verbose=True):
    img = readGrayImg(imgfolder, imgname)

    noiseAmplitude = 20
    imgNoise = randomNoiseImg(img.shape[0], img.shape[1], noiseAmplitude)

    mi = computeMutualInformation(img, imgNoise)

    if verbose:
        print("a) the MI of the image and 20-noise is [{}].".format(mi))

def main():
    print("2.1:")
    q2a()
    q2b()
    q2c()
    q2d()

    print("2.2")
    q22a()

if __name__=="__main__":
    main()