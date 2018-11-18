import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


def displayData(X, example_width=None):
    """
    DISPLAYDATA Display 2D data in a nice grid
    [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
    stored in X in a nice grid. It returns the figure handle h and the
    displayed array if requested.
    """
    # % Set example_width automatically if not passed in
    if example_width is None or example_width == '':
        example_width = np.round(np.sqrt(X[:, 2].size))

    m, n = X.shape

    # NOTE: np.round(), np.floor(), np.ceil(), all of these return float nums
    # Set example_width automatically if not passed in
    example_width = np.round(np.sqrt(n)).astype(int)
    example_height = (n / example_width).astype(int)

    # Compute the number of items to display
    display_rows = np.floor(np.sqrt(m)).astype(int)
    display_cols = np.ceil(m / display_rows).astype(int)

    # Between images padding
    pad = 1
    column = 1

    # Setup blank display
    display_array = - np.ones(  # % Compute rows, cols
        (pad + display_rows * (example_height + pad),
         pad + display_cols * (example_width + pad)))
    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex > m:
                break

            # Copy the patch
            # Get the max value of the patch
            max_val = np.max(np.abs(X[curr_ex]))
            display_array[pad + j * (example_height + pad) + np.arange(example_height),
                          pad + i * (example_width + pad) + np.arange(example_width)[:, np.newaxis]] = \
                X[curr_ex].reshape((example_height, example_width)) / max_val
            curr_ex += 1

        if curr_ex > m:
            break

    # Display image
    plt.figure()
    plt.imshow(display_array, cmap='gray', extent=[-1, 1, -1, 1])
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    data_dict = sio.loadmat('ex3data1.mat')  # % training data stored in arrays X, y
    X = data_dict['X']
    y = data_dict['y']
    m = X[:, 1].size

    # % Randomly select 100 data points to display
    rand_indices = np.random.permutation(m)
    sel = X[rand_indices[:100], :]

    displayData(sel)
