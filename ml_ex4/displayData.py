import numpy as np
import matplotlib.pyplot as plt


def displayData(X, example_width=None):
    """
    %DISPLAYDATA Display 2D data in a nice grid
    [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
    stored in X in a nice grid. It returns the figure handle h and the
    displayed array if requested.

    :param X:
    :return:
    """
    # m: NO. of digits in display
    # n: NO. of pixels in one example
    m, n = X.shape

    if example_width is None:
        example_width = int(np.round(np.sqrt(n)))

    # Compute rows, cols
    example_height = int(n / example_width)

    # Compute num of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    # Between blank padding
    pad = 1
    padding_row = -np.ones((1, pad+display_cols*(example_width+pad)))
    sub_padding_column = -np.ones((example_height, 1))

    # Copy each example into a patch on the display array
    display_array = padding_row

    curr_ex = 0
    for i in range(display_rows):
        temp_array = sub_padding_column
        for j in range(display_cols):
            curr_max = np.max(np.abs(X[curr_ex]))
            curr_ex_array = X[curr_ex].reshape(example_height, example_width).T / curr_max
            temp_array = np.block([temp_array, curr_ex_array, sub_padding_column])
            curr_ex += 1
        display_array = np.block([[display_array], [temp_array], [padding_row]])

    plt.figure()
    plt.imshow(display_array, cmap='gray', extent=[-1, 1, -1, 1])
    plt.axis('off')
    plt.show()
