import matplotlib.pyplot as plt


def plotData(x, y):
    """
    PLOTDATA Plots the data points x and y into a new figure
    PLOTDATA(x,y) plots the data points and gives the figure axes labels of
    population and profit.
    """
    plt.figure(1)
    plt.scatter(x, y, c='red', marker='x', label='Training data')

    # Plot Configuration
    plt.xlabel('population')
    plt.ylabel('profit')

    # plt.show()
