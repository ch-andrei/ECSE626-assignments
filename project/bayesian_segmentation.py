import numpy as np
import cv2
from numba import jit
import datetime

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
img_name3_trimap = "gandalfTrimap.png"

########################################################################################################################

epsilon_precision = 1e-6 # for precision

########################################################################################################################

# configuration
img2read = img_name3
trimap2read = img_name3_trimap
rescale = 1

########################################################################################################################


def imread(imgname, img_folder=img_folder, rescale=rescale, grayscale=False):
    img_name = "{}/{}".format(img_folder, imgname)
    img = cv2.imread(img_name) if not grayscale else cv2.imread(img_name, 0)
    return cv2.resize(img, (0, 0), fx=rescale, fy=rescale)


def imshow(imgname, img):
    cv2.imshow(imgname, img.astype(np.uint8))


def img2rgbArray(img):
    w, h = img.shape[:2]
    ar = img.ravel().reshape((w * h, 3))
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


def gaussian_2d(win_size=2, sigma=8):
    x, y = np.ogrid[-win_size:win_size+1, -win_size:win_size+1]
    g = np.exp(-((x * x + y * y) / (2.0 * sigma ** 2)))
    g /= g.sum()
    g /= g.max()
    g[g < epsilon_precision] = 0
    return g

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
    F_max = np.zeros(dim)
    B_max = np.zeros(dim)
    a_max = 0
    likelihood_max = -np.inf

    I = np.eye(dim)
    sigma_C_inv = 1.0 / (sigma_C * sigma_C)

    fg_clusters = fg_means.shape[0]
    bg_clusters = bg_means.shape[0]

    for fg_cluster_num in range(fg_clusters):
        fg_mean = fg_means[fg_cluster_num]
        fg_covar = fg_covars[fg_cluster_num]

        fg_covar_inv = np.linalg.inv(fg_covar)

        for bg_cluster_num in range(bg_clusters):
            bg_mean = bg_means[bg_cluster_num]
            bg_covar = bg_covars[bg_cluster_num]

            bg_covar_inv = np.linalg.inv(bg_covar)
            alpha = alpha_init

            likelihood_prev = -1e100
            likelihood_delta = np.inf
            iteration_num = 0
            while iteration_num < max_num_iterations and likelihood_delta > min_delta_likelihood:
                # structure the problem as Ax=b as per the paper, then compute x = solve(A,b)

                # build A
                A = np.zeros((6, 6))
                A[0:3, 0:3] = fg_covar_inv + I * alpha ** 2 * sigma_C_inv
                A[0:3, 3:6] = I * alpha * (1 - alpha) * sigma_C_inv
                A[3:6, 0:3] = A[0:3, 3:6]
                A[3:6, 3:6] = bg_covar_inv + I * (1 - alpha) ** 2 * sigma_C_inv

                # build b
                b = np.zeros((6, 1))
                b[0:3] = np.atleast_2d(fg_covar_inv @ fg_mean + C * (alpha) * sigma_C_inv).T
                b[3:6] = np.atleast_2d(bg_covar_inv @ bg_mean + C * (1 - alpha) * sigma_C_inv).T

                # solve for F and B values
                X = np.linalg.solve(A, b)
                F = np.maximum(np.minimum(1, X[0:3]), 0)
                B = np.maximum(np.minimum(1, X[3:6]), 0)

                # solve for alpha using the values for F and B
                alpha = (((np.atleast_2d(C).T - B).T @ (F - B)) / (np.sum((F - B) * (F - B)) + epsilon_precision))[0, 0]
                alpha = np.maximum(0, np.minimum(1, alpha))

                # # calculate likelihoods
                L_C = -np.sum((np.atleast_2d(C).T - alpha * F - (1 - alpha) * B) ** 2) * sigma_C_inv
                L_F = (-((F - np.atleast_2d(fg_mean).T).T @ fg_covar_inv @ (F - np.atleast_2d(fg_mean).T)) / 2)[0, 0]
                L_B = (-((B - np.atleast_2d(bg_mean).T).T @ bg_covar_inv @ (B - np.atleast_2d(bg_mean).T)) / 2)[0, 0]
                likelihood = (L_C + L_F + L_B)

                if likelihood_max < likelihood:
                    likelihood_max = likelihood
                    F_max = F.ravel()
                    B_max = B.ravel()
                    a_max = alpha

                likelihood_delta = np.abs(likelihood_prev - likelihood)
                iteration_num += 1

    return F_max, B_max, a_max

def bayesian_matting(img, trimap, win_radius=12, std_dev_c=0.1):
    img = img.astype(np.float)
    img /= 255

    min_num_samples = win_radius
    h, w, c = img.shape[:3]

    # compute gaussian weights; this will be used later
    gaussian_weights = gaussian_2d(win_radius, sigma=8)
    erode_kernel = np.ones((3, 3))

    # print(img.shape)
    # print(trimap.shape)

    # set up the masks
    foreground_mask = trimap == 255
    background_mask = trimap == 0
    unknown_mask = np.logical_not(np.logical_or(foreground_mask, background_mask))

    foreground_pixels = img * foreground_mask[:, :, np.newaxis]
    background_pixels = img * background_mask[:, :, np.newaxis]

    # cv2.imshow("foreground_pixels", foreground_pixels)
    # cv2.imshow("background_pixels", background_pixels)
    # cv2.waitKey()

    # alpha mask, 0 at background, 1 at foreground
    alpha_mask = np.zeros((h, w), np.float)
    alpha_mask[foreground_mask] = 1.0
    alpha_mask[unknown_mask] = np.nan

    # def remove_nans_window(pixels, weights):
    #     inds = np.nan_to_num(weights) > 0
    #     return inds.sum(), pixels[inds], weights[inds]

    num_unknown_pixels = np.sum(unknown_mask)
    pixel = 0
    while pixel < num_unknown_pixels:
        # TODO: erode unknown_mask to work iteratively on edges and not all pixels
        unknown_pixels = unknown_mask
        # unknown_mask_eroded = cv2.erode(unknown_mask.astype(np.uint8), erode_kernel, iterations=1)
        # unknown_pixels = np.logical_and(np.logical_not(unknown_mask_eroded), unknown_mask)

        Y, X = np.nonzero(unknown_pixels)

        for i, j in zip(Y, X):
            C = img[i, j]

            # window of alpha values around pixel (i, j)
            alpha_window = get_window(alpha_mask[..., np.newaxis], i, j, win_radius)[..., 0]

            # window of foreground values and weights around pixel (i, j)
            fg_window = img2rgbArray(get_window(foreground_pixels, i, j, win_radius))
            fg_weights = (alpha_window * alpha_window * gaussian_weights).flatten()
            inds = np.nan_to_num(fg_weights) > 0
            count_fg_samples = inds.sum()
            fg_window = fg_window[inds]
            fg_weights = fg_weights[inds]

            # window of background values and weights around pixel (i, j)
            bg_window = img2rgbArray(get_window(background_pixels, i, j, win_radius))
            bg_weights = ((1 - alpha_window) * (1 - alpha_window) * gaussian_weights).flatten()
            inds = np.nan_to_num(bg_weights) > 0
            count_bg_samples = inds.sum()
            bg_window = bg_window[inds]
            bg_weights = bg_weights[inds]

            # skip this iteration, if not enough samples
            if count_fg_samples < min_num_samples or count_bg_samples < min_num_samples:
                continue

            # run bouman-orchard clustering
            fg_cluster = ClusterTree(fg_window, fg_weights)
            bg_cluster = ClusterTree(bg_window, bg_weights)

            # get means and covars of the clusters
            fg_means, fg_covars = fg_cluster.get_cluster_stats()
            bg_means, bg_covars = bg_cluster.get_cluster_stats()

            # skip this iteration, if not enough clusters
            if fg_means.shape[0] < 1 or bg_means.shape[0] < 1:
                continue

            alpha_init = np.nanmean(alpha_window)
            # print("Before {}".format(str(datetime.datetime.now().time())))
            F, B, alpha = iterate_maximize(C, std_dev_c, fg_means, fg_covars, bg_means, bg_covars, alpha_init, 100, 1e-1)
            # print("After {}".format(str(datetime.datetime.now().time())))

            foreground_pixels[i, j] = F.ravel()
            background_pixels[i, j] = B.ravel()
            alpha_mask[i, j] = alpha
            unknown_mask[i, j] = 0

            if pixel % 10 == 0:
                print("\rProgress {}/{}".format(pixel, num_unknown_pixels), end='')

            pixel += 1

    return alpha_mask


def main():
    img = imread(imgname=img2read)
    trimap = imread(imgname=trimap2read, grayscale=True)
    alpha = bayesian_matting(img, trimap)

    imshow("img", img)
    imshow("alpha", alpha*255)
    cv2.waitKey()

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





    # cv2.waitKey()


if __name__=="__main__":
    main()
