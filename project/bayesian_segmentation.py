import numpy as np
import cv2
from numba import jit
import time

from project.bouman_orchard_clustering import *

########################################################################################################################

do_verbose = False
do_save_segmented = True
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

img1 = ("gandalf.png", "gandalfTrimap.png")
img2 = ("jomojo.JPG", "jomojo_trimap.png")
img3 = ("knockout01.png", "knockout01_trimap.png")
img4 = ("jomojo1.png", "jomojo1_trimap.png")
img5 = ("girl.png", "girl_trimap.png")
img6 = ("tower.png", "tower_trimap.png")

########################################################################################################################

epsilon_precision = 1e-6 # for precision

########################################################################################################################

# configuration
img2read, trimap2read = img3

########################################################################################################################

# configuration rescaling
rescale_compute = 1
rescale_imshow = 1

########################################################################################################################


def imread(imgname, img_folder=img_folder, rescale=rescale_compute, grayscale=False, use_nearest=False):
    img_name = "{}/{}".format(img_folder, imgname)
    img = cv2.imread(img_name) if not grayscale else cv2.imread(img_name, 0)
    if not rescale == 1:
        img = cv2.resize(img, (0, 0), fx=rescale, fy=rescale,
                   interpolation=cv2.INTER_NEAREST if use_nearest else cv2.INTER_CUBIC)
    return img


def imshow(imgname, img, tag="", undo_rescale_compute=True, imwrite=True):
    rescale = rescale_imshow
    if undo_rescale_compute:
        rescale /= rescale_compute
    resized_img = cv2.resize(img.astype(np.uint8), (0, 0),
                                 fx=rescale,
                                 fy=rescale)
    if imwrite:
        imsave(imgname, img, tag)
    cv2.imshow(tag, resized_img)


def imsave(imgname, img, tag=""):
    if not tag == "":
        imgname_out = "{}-{}.png".format(imgname.split('.')[0], tag)
    else:
        imgname_out = "{}.png".format(imgname.split('.')[0])
    cv2.imwrite("{}/{}".format(img_result_folder, imgname_out), img.astype(np.uint8))


def img2rgbArray(img):
    w, h, c = img.shape
    ar = img.ravel().reshape((w * h, c))
    return ar


def preprocess_image(img):
    x = img2rgbArray(img)
    x = x.transpose() # transpose X to be N x d
    return x


def read_preprocess_image(imgname=img2read):
    img = imread(imgname)
    x = preprocess_image(img)
    return img, x


def check_nan(val):
    return np.isnan(val) or np.isinf(val) or np.isneginf(val)


########################################################################################################################


def gaussian_2d(window_radius, sigma):
    x, y = np.ogrid[-window_radius: window_radius + 1, -window_radius: window_radius + 1]
    # x = x.astype(np.float) / window_radius
    # y = y.astype(np.float) / window_radius
    g = np.exp(-((x * x + y * y) / (2.0 * sigma ** 2)))
    g /= g.sum()
    g /= g.max()
    g[g < epsilon_precision] = 0
    return g


class GaussianWeightsDict:
    def __init__(self, sigma=8):
        self.weights = {}
        self.sigma = sigma

    def get(self, window_radius):
        if window_radius not in self.weights:
            self.weights[window_radius] = gaussian_2d(window_radius, self.sigma)
        return self.weights[window_radius]


@jit(nopython=True, cache=True)
def get_window(values, i, j, win_radius):
    h, w, c = values.shape

    size = win_radius * 2 + 1
    win = np.zeros((size, size, values.shape[2]))

    i_min = max(i - win_radius, 0)
    j_min = max(j - win_radius, 0)
    i_max = min(i + win_radius, h)
    j_max = min(j + win_radius, w)

    i_min_w = win_radius - (i - i_min)
    j_min_w = win_radius - (j - j_min)
    i_max_w = win_radius + (i_max - i)
    j_max_w = win_radius + (j_max - j)

    win[i_min_w: i_max_w, j_min_w: j_max_w] = values[i_min: i_max, j_min: j_max]

    return win


@jit(nopython=True, cache=True)
def iterate_maximize(C, sigma_C,
                     fg_means, fg_covars,
                     bg_means, bg_covars,
                     alpha_init,
                     max_num_iterations,
                     min_delta_likelihood
                     ):
    dim = 3

    # initialize outputs
    F_maxlike = np.zeros(0)
    B_maxlike = np.zeros(0)
    a_maxlike = 0
    likelihood_max = -np.inf

    I = np.eye(dim)
    sigma_C_inv = 1.0 / (sigma_C * sigma_C)

    for fg_cluster_num in range(fg_means.shape[0]):
        fg_mean = fg_means[fg_cluster_num]
        fg_covar = fg_covars[fg_cluster_num]

        fg_covar_inv = np.linalg.inv(fg_covar)

        for bg_cluster_num in range(bg_means.shape[0]):
            bg_mean = bg_means[bg_cluster_num]
            bg_covar = bg_covars[bg_cluster_num]

            bg_covar_inv = np.linalg.inv(bg_covar)
            alpha = alpha_init

            likelihood_prev = -1e100
            likelihood_delta = np.inf
            iteration_num = 0
            while iteration_num < max_num_iterations and likelihood_delta > min_delta_likelihood:
                # build A
                A = np.zeros((6, 6))
                A[0:3, 0:3] = fg_covar_inv + I * alpha ** 2 * sigma_C_inv
                A[0:3, 3:6] = I * alpha * (1 - alpha) * sigma_C_inv
                A[3:6, 0:3] = A[0:3, 3:6]
                A[3:6, 3:6] = bg_covar_inv + I * (1 - alpha) ** 2 * sigma_C_inv

                # build b
                b = np.zeros((1, 6))
                b[0, 0:3] = fg_covar_inv @ fg_mean + C * (alpha) * sigma_C_inv
                b[0, 3:6] = bg_covar_inv @ bg_mean + C * (1 - alpha) * sigma_C_inv

                # solve for F and B values
                X = np.maximum(0, np.minimum(1, np.linalg.solve(A, b.T)))
                F = X[0:3].flatten()
                B = X[3:6].flatten()

                # solve for alpha using the values for F and B
                alpha = ((C - B) @ (F - B)) / (np.sum((F - B) * (F - B)) + epsilon_precision)
                alpha = np.maximum(0, np.minimum(1, alpha))

                # calculate likelihoods
                L_C = -np.sum((C - alpha * F - (1 - alpha) * B) ** 2) * sigma_C_inv
                L_F = -(F - fg_mean) @ fg_covar_inv @ np.atleast_2d(F - fg_mean).T / 2
                L_B = -(B - bg_mean) @ bg_covar_inv @ np.atleast_2d(B - bg_mean).T / 2
                likelihood = (L_C + L_F + L_B)[0]

                if likelihood > likelihood_max:
                    likelihood_max = likelihood
                    F_maxlike = F.flatten()
                    B_maxlike = B.flatten()
                    a_maxlike = alpha

                likelihood_delta = np.abs(likelihood_prev - likelihood)
                likelihood_prev = likelihood
                iteration_num += 1

            # print("ITERATIONS", iteration_num)

    return F_maxlike, B_maxlike, a_maxlike


def bayesian_matting(img, trimap,
                     win_radius_min=5,
                     win_radius_max=25,
                     min_num_samples=20,
                     std_dev_c=0.01,
                     max_sweeps=100,
                     max_iterations_per_solve=100,
                     solve_convergence_threshold=1e-3,
                     shuffle_indices=False
                     ):
    print("Running bayesian matting on {}...".format(img2read))

    img = img.astype(np.float)
    img /= 255

    h, w, c = img.shape[:3]

    # compute gaussian weights; this will be used later
    gaussians = GaussianWeightsDict()
    erode_kernel = np.ones((3, 3))

    # set up the masks
    foreground_mask = trimap == 255
    background_mask = trimap == 0
    unknown_mask = np.logical_not(np.logical_or(foreground_mask, background_mask))

    foreground_pixels = img * foreground_mask[:, :, np.newaxis]
    background_pixels = img * background_mask[:, :, np.newaxis]

    # alpha mask, 0 at background, 1 at foreground
    alpha_mask = np.zeros((h, w), np.float)
    alpha_mask[foreground_mask] = 1.0
    alpha_mask[unknown_mask] = np.nan

    num_unknown_pixels = np.sum(unknown_mask)
    pixel = 0
    sweep = 0
    while pixel < num_unknown_pixels and sweep < max_sweeps:

        Y, X = np.nonzero(unknown_mask)

        if shuffle_indices:
            p = np.random.permutation(len(Y))
            Y = Y[p]
            X = X[p]

        pixels_per_sweep = 0
        print("[Sweep{}] attempting to solve {} pixels.".format(sweep, len(Y)))

        for i, j in zip(Y, X):
            C = img[i, j]

            # collect samples around pixel (i, j)
            # start with smallest window size then increase it until enough samples are collected
            count_fg_samples = 0
            count_bg_samples = 0
            alpha_window = None
            fg_window = None; bg_window = None
            fg_weights = None; bg_weights = None

            sample_failed = False
            win_radius = win_radius_min
            while count_fg_samples <= min_num_samples or count_bg_samples <= min_num_samples:
                # compute gaussian weights
                gaussian_weights = gaussians.get(win_radius)

                # window of alpha values around pixel (i, j)
                alpha_window = get_window(alpha_mask[..., np.newaxis], i, j, win_radius)[..., 0]

                # window of foreground values and weights around pixel (i, j)
                fg_window = img2rgbArray(get_window(foreground_pixels, i, j, win_radius))
                fg_weights = (alpha_window ** 2 * gaussian_weights).flatten()
                inds = np.nan_to_num(fg_weights) > 0
                count_fg_samples = inds.sum()
                fg_window = fg_window[inds]
                fg_weights = fg_weights[inds]

                # window of background values and weights around pixel (i, j)
                bg_window = img2rgbArray(get_window(background_pixels, i, j, win_radius))
                bg_weights = ((1 - alpha_window) ** 2 * gaussian_weights).flatten()
                inds = np.nan_to_num(bg_weights) > 0
                count_bg_samples = inds.sum()
                bg_window = bg_window[inds]
                bg_weights = bg_weights[inds]

                # increment for next iteration
                win_radius += 1

                # print(count_fg_samples, count_bg_samples, min_num_samples, alpha_window.shape)

                # skip this pixel if not enough samples were collected even for max window size
                if win_radius > win_radius_max + np.sqrt(sweep):
                    sample_failed = True
                    break

            if sample_failed:
                continue

            # run bouman-orchard clustering and get means and covars of the clusters
            fg_means, fg_covars = ClusterTree(fg_window, fg_weights, sigma_C=std_dev_c).get_cluster_stats()
            bg_means, bg_covars = ClusterTree(bg_window, bg_weights, sigma_C=std_dev_c).get_cluster_stats()

            # solve for F, B, alpha
            F, B, alpha = iterate_maximize(C, std_dev_c,
                                           fg_means, fg_covars,
                                           bg_means, bg_covars,
                                           np.nanmean(alpha_window),
                                           max_iterations_per_solve,
                                           solve_convergence_threshold)

            alpha_mask[i, j] = alpha
            foreground_pixels[i, j] = F
            background_pixels[i, j] = B
            unknown_mask[i, j] = 0

            if pixels_per_sweep % 50 == 0:
                print("\rProgress {}/{}".format(pixel + pixels_per_sweep, num_unknown_pixels), end='')

            if pixels_per_sweep % 1000 == 0:
                imsave(img2read, alpha_mask * 255, "alpha{}".format(pixel + pixels_per_sweep))
                imshow("alpha", alpha_mask * 255, imwrite=False)
                cv2.waitKey(10)

            pixels_per_sweep += 1

        print("\n[Sweep{}] successfully solved {}/{} unknown pixels.".format(sweep, pixels_per_sweep, len(Y)))
        if pixels_per_sweep == 0:
            min_num_samples = max(1, min_num_samples - 1)
        pixel += pixels_per_sweep
        sweep += 1

    print("Bayesian matting complete.")

    alpha_mask = np.nan_to_num(alpha_mask)
    foreground_pixels = np.nan_to_num(foreground_pixels)
    background_pixels = np.nan_to_num(background_pixels)

    return alpha_mask, foreground_pixels, background_pixels


def do_matting_single_channel():
    img = imread(imgname=img2read)

    reds = np.repeat(np.atleast_3d(img[..., 0]), 3, axis=2)
    blues = np.repeat(np.atleast_3d(img[..., 1]), 3, axis=2)
    greens = np.repeat(np.atleast_3d(img[..., 2]), 3, axis=2)
    trimap = imread(imgname=trimap2read, grayscale=True)

    time1 = time.time()
    alpha_red, fg, bg = bayesian_matting(reds, trimap)
    alpha_blue, fg, bg = bayesian_matting(blues, trimap)
    alpha_green, fg, bg = bayesian_matting(greens, trimap)
    delta_time = time.time() - time1
    print("Took {} seconds".format(delta_time))

    fg = np.zeros_like(img)
    fg[..., 0] = alpha_red * img[..., 0]
    fg[..., 1] = alpha_blue * img[..., 1]
    fg[..., 2] = alpha_green * img[..., 2]

    bg = np.zeros_like(img)
    bg[..., 0] = (1 - alpha_red) * img[..., 0]
    bg[..., 1] = (1 - alpha_blue) * img[..., 1]
    bg[..., 2] = (1 - alpha_green) * img[..., 2]

    imshow(img2read, img, imwrite=False)
    imshow(img2read, alpha_red*255, "alpha_red")
    imshow(img2read, alpha_blue*255, "alpha_blue")
    imshow(img2read, alpha_green*255, "alpha_green")
    imshow(img2read, fg, "fg")
    imshow(img2read, bg, "bg")
    cv2.waitKey()


def do_matting_rgb():
    img = imread(imgname=img2read)
    trimap = imread(imgname=trimap2read, grayscale=True, use_nearest=True)

    time1 = time.time()
    alpha, fg, bg = bayesian_matting(img, trimap)
    delta_time = time.time() - time1
    print("Took {} seconds".format(delta_time))

    imshow(img2read, img, imwrite=False)
    imshow(img2read, alpha*255, tag="alpha")
    imshow(img2read, (alpha[..., np.newaxis]) * img, tag="fg")
    imshow(img2read, (1.0-alpha[..., np.newaxis]) * img, tag="bg")
    imshow(img2read, fg * 255, tag="fg_bayes")
    imshow(img2read, bg * 255, tag="bg_bayes")
    cv2.waitKey()

def main():
    do_matting_rgb()

    # cov = np.arange(9).reshape((3,3))
    # a = np.arange(3)
    # print(a)
    # print(cov)
    # print(cov @ a)

    # def show_tree(cluster_tree, name):
    #     img_mask = np.zeros(h * w, np.float)
    #     cluster_nodes = cluster_tree.get_leaf_nodes()
    #     for index, cluster_node in enumerate(cluster_nodes):
    #         mask = cluster_node.get_cluster_mask(img)
    #         img_mask[mask] = index
    #     img_mask /= img_mask.max()
    #     img_mask *= 255
    #     img_mask = img_mask.reshape((h, w))
    #     imshow("mask{}".format(name), img_mask)

if __name__=="__main__":
    main()
