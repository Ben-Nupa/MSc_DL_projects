import matplotlib.pyplot as plt
import matplotlib.patches as patches
# %matplotlib inline
import numpy as np
from keras.utils import np_utils

# On some implementations of matplotlib, you may need to change this value
IMAGE_SIZE = 72


def generate_a_drawing(figsize, U, V, noise=0.0, return_both=False):
    """@:param return_both : Set to True to generate pairs of images, where one image has noise with random amplitude,
    and the second image has the same content but without the noise."""
    fig = plt.figure(figsize=(figsize, figsize))
    ax = plt.subplot(111)
    plt.axis('Off')
    ax.set_xlim(0, figsize)
    ax.set_ylim(0, figsize)
    ax.fill(U, V, "k")
    fig.canvas.draw()
    imdata = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)[::3].astype(np.float32)
    if not return_both:
        imdata = imdata + noise * np.random.random(imdata.size)
        plt.close(fig)
        return imdata
    else:
        noisy_imdata = imdata + noise * np.random.random(imdata.size)
        plt.close(fig)
        return imdata, noisy_imdata


def generate_a_rectangle(noise=0.0, free_location=False, return_both=False):
    """@:param return_both : Set to True to generate pairs of images, where one image has noise with random amplitude,
    and the second image has the same content but without the noise."""
    figsize = 1.0
    U = np.zeros(4)
    V = np.zeros(4)
    if free_location:
        corners = np.random.random(4)
        top = max(corners[0], corners[1])
        bottom = min(corners[0], corners[1])
        left = min(corners[2], corners[3])
        right = max(corners[2], corners[3])
    else:
        side = (0.3 + 0.7 * np.random.random()) * figsize
        top = figsize / 2 + side / 2
        bottom = figsize / 2 - side / 2
        left = bottom
        right = top
    U[0] = U[1] = top
    U[2] = U[3] = bottom
    V[0] = V[3] = left
    V[1] = V[2] = right
    return generate_a_drawing(figsize, U, V, noise, return_both=return_both)


def generate_a_disk(noise=0.0, free_location=False, return_both=False):
    """@:param return_both : Set to True to generate pairs of images, where one image has noise with random amplitude,
    and the second image has the same content but without the noise."""
    figsize = 1.0
    if free_location:
        center = np.random.random(2)
    else:
        center = (figsize / 2, figsize / 2)
    radius = (0.3 + 0.7 * np.random.random()) * figsize / 2
    N = 50
    U = np.zeros(N)
    V = np.zeros(N)
    i = 0
    for t in np.linspace(0, 2 * np.pi, N):
        U[i] = center[0] + np.cos(t) * radius
        V[i] = center[1] + np.sin(t) * radius
        i = i + 1
    return generate_a_drawing(figsize, U, V, noise, return_both=return_both)


def generate_a_triangle(noise=0.0, free_location=False, return_both=False):
    """@:param return_both : Set to True to generate pairs of images, where one image has noise with random amplitude,
    and the second image has the same content but without the noise."""
    figsize = 1.0
    if free_location:
        U = np.random.random(3)
        V = np.random.random(3)
    else:
        size = (0.3 + 0.7 * np.random.random()) * figsize / 2
        middle = figsize / 2
        U = (middle, middle + size, middle - size)
        V = (middle + size, middle - size, middle - size)
    if not return_both:
        imdata = generate_a_drawing(figsize, U, V, noise, return_both=False)
        return [imdata, [U[0], V[0], U[1], V[1], U[2], V[2]]]
    else:
        imdata, noisy_imdata = generate_a_drawing(figsize, U, V, noise, return_both=True)
        return [imdata, noisy_imdata, [U[0], V[0], U[1], V[1], U[2], V[2]]]


def generate_dataset_classification(nb_samples, noise=0.0, free_location=False):
    # Getting im_size:
    im_size = generate_a_rectangle().shape[0]
    X = np.zeros([nb_samples, im_size])
    Y = np.zeros(nb_samples)
    print('Creating data:')
    for i in range(nb_samples):
        if i % 100 == 0:
            print(i)
        category = np.random.randint(3)
        if category == 0:
            X[i] = generate_a_rectangle(noise, free_location)
        elif category == 1:
            X[i] = generate_a_disk(noise, free_location)
        else:
            [X[i], V] = generate_a_triangle(noise, free_location)
        Y[i] = category
    X = (X + noise) / (255 + 2 * noise)
    return [X, Y]


def generate_test_set_classification():
    np.random.seed(42)
    [X_test, Y_test] = generate_dataset_classification(300, 20, True)
    Y_test = np_utils.to_categorical(Y_test, 3)
    return [X_test, Y_test]


def generate_dataset_regression(nb_samples, noise=0.0):
    # Getting im_size:
    im_size = generate_a_triangle()[0].shape[0]
    X = np.zeros([nb_samples, im_size])
    Y = np.zeros([nb_samples, 6])
    print('Creating data:')
    for i in range(nb_samples):
        if i % 100 == 0:
            print(i)
        [X[i], Y[i]] = generate_a_triangle(noise, True)
    X = (X + noise) / (255 + 2 * noise)
    return [X, Y]


def visualize_prediction(x, y):
    fig, ax = plt.subplots(figsize=(5, 5))
    I = x.reshape((IMAGE_SIZE, IMAGE_SIZE))
    ax.imshow(I, extent=[-0.15, 1.15, -0.15, 1.15], cmap='gray')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    xy = y.reshape(3, 2)
    tri = patches.Polygon(xy, closed=True, fill=False, edgecolor='r', linewidth=5, alpha=0.5)
    ax.add_patch(tri)

    plt.show()


def generate_test_set_regression():
    np.random.seed(42)
    [X_test, Y_test] = generate_dataset_regression(300, 20)
    return [X_test, Y_test]


def generate_dataset_denoising(nb_samples, free_location=False):
    # Getting im_size:
    im_size = generate_a_rectangle().shape[0]
    X = np.zeros([nb_samples, im_size])
    X_noisy = np.zeros([nb_samples, im_size])
    noise = []
    print('Creating data:')
    for i in range(nb_samples):
        if i % 100 == 0:
            print(i)
        category = np.random.randint(3)
        noise.append(100 * np.random.random())
        if category == 0:
            X[i], X_noisy[i] = generate_a_rectangle(noise[-1], free_location, return_both=True)
        elif category == 1:
            X[i], X_noisy[i] = generate_a_disk(noise[-1], free_location, return_both=True)
        else:
            X[i], X_noisy[i], _ = generate_a_triangle(noise[-1], free_location, return_both=True)
    X = X / 255
    X_noisy = (X_noisy + np.min(noise)) / (255 + np.max(noise) + np.min(noise))
    return X.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1), X_noisy.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)


def generate_test_set_denoising():
    return generate_dataset_denoising(300, False)


def visualize_denoising(x_noisy, x_real, x_reconstruced=None):
    plt.figure(figsize=(15, 15))

    plt.subplot(1, 3, 1)
    plt.title("Real drawing")
    plt.imshow(x_real.reshape((IMAGE_SIZE, IMAGE_SIZE)), cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Noisy drawing")
    plt.imshow(x_noisy.reshape((IMAGE_SIZE, IMAGE_SIZE)), cmap='gray')

    if x_reconstruced is not None:
        plt.subplot(1, 3, 3)
        plt.title("Reconstructed drawing")
        plt.imshow(x_reconstruced.reshape((IMAGE_SIZE, IMAGE_SIZE)), cmap='gray')

    plt.show()


if __name__ == '__main__':
    IMAGE_SIZE = 100

    im = generate_a_rectangle(10, True)
    plt.imshow(im.reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
    plt.show()

    im = generate_a_disk(10)
    plt.imshow(im.reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
    plt.show()

    [im, v] = generate_a_triangle(20, False)
    plt.imshow(im.reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
    plt.show()
