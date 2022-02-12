import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random


def init_centroids(num_clusters, image):
    """
    Initialize a `num_clusters` x image_shape[-1] nparray to RGB
    values of randomly chosen pixels of `image`

    Parameters
    ----------
    num_clusters : int
        Number of centroids/clusters
    image : nparray
        (H, W, C) image represented as an nparray

    Returns
    -------
    centroids_init : nparray
        Randomly initialized centroids
    """

    # *** START YOUR CODE ***
    # raise NotImplementedError('init_centroids function not implemented')

    sirina, visina = image[:,:,0].shape

    centroids_init = np.zeros((num_clusters, 3))

    for i in range(num_clusters):
        x = random.randrange(sirina)
        y = random.randrange(visina)
        centroids_init[i] = image[x, y, :]

    # *** END YOUR CODE ***

    return centroids_init


def update_centroids(centroids, image, max_iter=30, print_every=10):
    """
    Carry out k-means centroid update step `max_iter` times

    Parameters
    ----------
    centroids : nparray
        The centroids stored as an nparray
    image : nparray
        (H, W, C) image represented as an nparray
    max_iter : int
        Number of iterations to run
    print_every : int
        Frequency of status update

    Returns
    -------
    new_centroids : nparray
        Updated centroids
    """

    # *** START YOUR CODE ***
    # raise NotImplementedError('update_centroids function not implemented')
    
    # Usually expected to converge long before `max_iter` iterations
    vremepocetka = os.times().elapsed
    sirina, visina = image[:,:,0].shape

    new_centroids = centroids

    for iteracija in range(max_iter):
        pre = new_centroids.copy()

        suma_centroida = np.zeros(new_centroids.shape)
        br_centroida = np.zeros((len(new_centroids), 1))

        for x in range(sirina):
            for y in range(visina):
                r, g, b = image[x,y,:]

                najblizi = new_centroids[0].copy()
                index_najblizeg = 0
                distanca_najblizeg = (new_centroids[0][0] - r) ** 2 + (new_centroids[0][1] - g) ** 2 + (new_centroids[0][2] - b) ** 2

                for i in range(1, len(new_centroids)):
                    d = (new_centroids[i][0] - r) ** 2 + (new_centroids[i][1] - g) ** 2 + (new_centroids[i][2] - b) ** 2
                    if distanca_najblizeg > d:
                        distanca_najblizeg = d
                        index_najblizeg = i
                        najblizi = new_centroids[i]
                
                suma_centroida[index_najblizeg] += [r, g, b]
                br_centroida[index_najblizeg] += 1

        for i in range(len(new_centroids)):
            if br_centroida[i] == 0:
                br_centroida[i] = 1
            new_centroids[i] = np.true_divide(suma_centroida[i], br_centroida[i])

        if iteracija % print_every == 0:
            print("Iteracija " + str(iteracija))

        if np.all((new_centroids - pre) == 0):
            print(f"Iteracija: {iteracija}, konacno: ")
            print(new_centroids)
            break

    # Loop over all centroids and store distances in `dist`
    # Find closest centroid and update `new_centroids`
    # Update `new_centroids`
    print("Vreme: ", os.times().elapsed - vremepocetka) 
    # *** END YOUR CODE ***

    return new_centroids


def update_image(image, centroids):
    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray

    Returns
    -------
    image : nparray
        Updated image
    """

    # *** START YOUR CODE ***

    sirina, visina = image[:, :, 0].shape

    c = np.zeros(image.shape)

    def dist(centroid, r, g, b):
        return np.sqrt((centroid[0] - r) ** 2 + (centroid[1] - g) ** 2 + (centroid[2] - b) ** 2)
    
    for x in range(sirina):
        for y in range(visina):
            r, g, b = image[x,y,:]

            najblizi = centroids[0].copy()
            index_najblizeg = 0
            distanca_najblizeg = (centroids[0][0] - r) ** 2 + (centroids[0][1] - g) ** 2 + (centroids[0][2] - b) ** 2

            for i in range(1, len(centroids)):
                d = (centroids[i][0] - r) ** 2 + (centroids[i][1] - g) ** 2 + (centroids[i][2] - b) ** 2
                if distanca_najblizeg > d:
                    distanca_najblizeg = d
                    index_najblizeg = i
                    najblizi = centroids[i]            

            image[x, y, :] = najblizi

    # raise NotImplementedError('update_image function not implemented')
    # Initialize `dist` vector to keep track of distance to every centroid
    # Loop over all centroids and store distances in `dist`
    # Find closest centroid and update pixel value in `image`
    # *** END YOUR CODE ***

    return image


def main(args):

    # Setup
    max_iter = args.max_iter
    print_every = args.print_every
    image_path_small = args.small_path
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0

    # Load small image
    image = np.copy(mpimg.imread(image_path_small))
    print('[INFO] Loaded small image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original small image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_small.png')
    plt.savefig(savepath, transparent=True, format='png', bbox_inches='tight')

    # Initialize centroids
    print('[INFO] Centroids initialized')
    centroids_init = init_centroids(num_clusters, image)

    # Update centroids
    print(25 * '=')
    print('Updating centroids ...')
    print(25 * '=')
    centroids = update_centroids(centroids_init, image, max_iter, print_every)

    # Load large image
    image = np.copy(mpimg.imread(image_path_large))
    image.setflags(write=1)
    print('[INFO] Loaded large image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original large image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    # Update large image with centroids calculated on small image
    print(25 * '=')
    print('Updating large image ...')
    print(25 * '=')
    image_clustered = update_image(image, centroids)

    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image_clustered)
    plt.title('Updated large image')
    plt.axis('off')
    savepath = os.path.join('.', 'updated_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    print('\nCOMPLETE')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default='./peppers-small.tiff',
                        help='Path to small image')
    parser.add_argument('--large_path', default='./peppers-large.tiff',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=250,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=16,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)
