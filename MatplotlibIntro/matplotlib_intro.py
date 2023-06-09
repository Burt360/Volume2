# matplotlib_intro.py
"""Python Essentials: Intro to Matplotlib.
Nathan Schill
Sec. 3
Thurs. Sept. 21, 2022
"""

import numpy as np
from matplotlib import pyplot as plt

# Problem 1
def var_of_means(n):
    """Create an (n x n) array of values randomly sampled from the standard
    normal distribution. Compute the mean of each row of the array. Return the
    variance of these means.

    Parameters:
        n (int): The number of rows and columns in the matrix.

    Returns:
        (float) The variance of the means of each row.
    """
    # nxn matrix of values randomly sampled from normal distribution
    A = np.random.normal(size=(n, n))

    # Variance of means of each row
    return np.var(np.mean(A, axis=1))

def prob1():
    """Create an array of the results of var_of_means() with inputs
    n = 100, 200, ..., 1000. Plot and show the resulting array.
    """

    # Array of var_of_means(n) for nxn matrices for n = 100, 200, ..., 1000
    x = list(range(100, 1001, 100))
    result = np.array([var_of_means(n) for n in x])

    # Create the plot.
    plt.plot(x, result)

    # Add title and axis labels.
    plt.title('Variance of means of rows for nxn matrices')
    plt.ylabel('Variance of means of rows')
    plt.xlabel('n')

    # Show the plot.
    plt.show()

# Problem 2
def prob2():
    """Plot the functions sin(x), cos(x), and arctan(x) on the domain
    [-2pi, 2pi]. Make sure the domain is refined enough to produce a figure
    with good resolution.
    """
    
    # Array of 100 points between -2Pi and 2Pi
    x = np.linspace(-2 * np.pi, 2 * np.pi, 100)

    # Plot f(x) over x for f as each of the given functions.
    pairs = ((np.sin, 'sin(x)'), (np.cos, 'cos(x)'), (np.arctan, 'arctan(x)'))
    [plt.plot(x, fn(x), label=name) for fn, name in pairs]

    # Add title, axis labels, and legend. 
    plt.title('sin(x), cos(x), arctan(x)')
    plt.legend()
    plt.ylabel('y')
    plt.xlabel('x')

    # Show the plot.
    plt.show() 


# Problem 3
def prob3():
    """Plot the curve f(x) = 1/(x-1) on the domain [-2,6].
        1. Split the domain so that the curve looks discontinuous.
        2. Plot both curves with a thick, dashed magenta line.
        3. Set the range of the x-axis to [-2,6] and the range of the
           y-axis to [-6,6].
    """
    
    # Left and right halves of domain (exlcuding x = 1, where f is discontinuous).
    x1 = np.linspace(-2, 0.99, 50)
    x2 = np.linspace(1.01, 6, 100)

    # Plot f(x) on each side of the domain.
    plt.plot(x1, 1/(x1-1), 'm--', linewidth=4)
    plt.plot(x2, 1/(x2-1), 'm--', linewidth=4)

    # Set axis bounds.
    plt.xlim(-2, 6)
    plt.ylim(-6, 6)

    # Add title and axis labels.
    plt.title('1/(x-1)')
    plt.ylabel('y')
    plt.xlabel('x')
    
    # Show the plot.
    plt.show()

# Problem 4
def prob4():
    """Plot the functions sin(x), sin(2x), 2sin(x), and 2sin(2x) on the
    domain [0, 2pi], each in a separate subplot of a single figure.
        1. Arrange the plots in a square grid of four subplots.
        2. Set the limits of each subplot to [0, 2pi]x[-2, 2].
        3. Give each subplot an appropriate title.
        4. Give the overall figure a title.
        5. Use the following line colors and styles.
              sin(x): green solid line.
             sin(2x): red dashed line.
             2sin(x): blue dashed line.
            2sin(2x): magenta dotted line.
    """
    # Create 50 points between 0 and 2PI.
    x = np.linspace(0, 2 * np.pi, 50)

    # Create 4 subplots, arranged 2x2.
    fig, axes = plt.subplots(2, 2)

    # Create plots and set titles.
    # sin(x)
    axes[0][0].plot(x, np.sin(x), 'g-')
    axes[0][0].set_title('sin(x)')

    # sin(2x)
    axes[0][1].plot(x, np.sin(2*x), 'r--')
    axes[0][1].set_title('sin(2x)')
    
    # 2sin(x)
    axes[1][0].plot(x, 2*np.sin(x), 'b--')
    axes[1][0].set_title('2sin(x)')

    # 2sin(2x)
    axes[1][1].plot(x, 2*np.sin(2*x), 'm:')
    axes[1][1].set_title('2sin(2x)')

    # Set axis limits.
    [axes[i][o].set_xlim(0, 2*np.pi) for i in range(2) for o in range(2)]
    [axes[i][o].set_ylim(-2, 2) for i in range(2) for o in range(2)]

    # Set axis labels.
    [axes[i][o].set_xlabel('x') for i in range(2) for o in range(2)]
    [axes[i][o].set_ylabel('y') for i in range(2) for o in range(2)]

    # Add title on the entire figure.
    fig.suptitle('Sine comparison')
    
    # Adjust spacing and show the plot.
    fig.tight_layout()
    plt.show()


# Problem 5
def prob5():
    """Visualize the data in FARS.npy. Use np.load() to load the data, then
    create a single figure with two subplots:
        1. A scatter plot of longitudes against latitudes. Because of the
            large number of data points, use black pixel markers (use "k,"
            as the third argument to plt.plot()). Label both axes.
        2. A histogram of the hours of the day, with one bin per hour.
            Label and set the limits of the x-axis.
    """
    # Load the data.
    data = np.load('FARS.npy')
    #print(data[:100,:])
    # Create 2 subplots, arranged 1x2.
    fig, axes = plt.subplots(1, 2)

    # Create map chart with equal aspect ratio, title, and axis labels.
    axes[0].plot(data[:, 1], data[:, 2], 'k,')
    axes[0].set_aspect('equal')
    axes[0].set_xlabel('Latitude')
    axes[0].set_ylabel('Longitude')
    axes[0].set_title('Locations of fatal accidents')

    # Create histogram with title and axis labels, and set axis limits.
    axes[1].hist(data[:, 0], bins=np.arange(0, 25))
    axes[1].set_xlim(0, 24)
    axes[1].set_xlabel('Hour of day')
    axes[1].set_ylabel('Number of fatal accidents')
    axes[1].set_title('Fatal accidents by hour of day')

    # Adjust spacing and show the plot.
    fig.tight_layout()
    fig.show()


# Problem 6
def prob6():
    """Plot the function g(x,y) = sin(x)sin(y)/xy on the domain
    [-2pi, 2pi]x[-2pi, 2pi].
        1. Create 2 subplots: one with a heat map of g, and one with a contour
            map of g. Choose an appropriate number of level curves, or specify
            the curves yourself.
        2. Set the limits of each subplot to [-2pi, 2pi]x[-2pi, 2pi].
        3. Choose a non-default color scheme.
        4. Include a color scale bar for each subplot.
    """
    # Create meshgrid.
    x = np.linspace(-2 * np.pi, 2 * np.pi, 100); y = x.copy()
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X)*np.sin(Y) / (X*Y)
    
    # Create heat map of g.
    plt.subplot(121)
    plt.pcolormesh(X, Y, Z, cmap='coolwarm')

    # Add colorbar, title, and axis labels.
    plt.colorbar()
    plt.title('Heat map of g')
    plt.xlabel('x')
    plt.ylabel('y')

    # Set axis limits.
    plt.xlim(-2*np.pi, 2*np.pi)
    plt.ylim(-2*np.pi, 2*np.pi)

    ########################################

    # Create contour map of g.
    plt.subplot(122)
    plt.contour(X, Y, Z, 10, cmap='coolwarm')

    # Add colorbar, title, and axis labels.
    plt.colorbar()
    plt.title('Contour map of g')
    plt.xlabel('x')
    plt.ylabel('y')

    # Set axis limits.
    plt.xlim(-2*np.pi, 2*np.pi)
    plt.ylim(-2*np.pi, 2*np.pi)

    # Add title on the entire figure.
    plt.suptitle('g(x) = sin(x)sin(y)/xy')

    # Adjust spacing and show the plot.
    plt.tight_layout()
    plt.show()







"""
### Additional material ###

from matplotlib.animation import FuncAnimation
def sine_animation():
    # Calculate the data to be animated.
    x = np.linspace(0, 2*np.pi, 200)[:-1]
    y = np.sin(x)
    # Create a figure and set the window boundaries of the axes.
    fig = plt.figure()
    plt.xlim(0, 2*np.pi)
    plt.ylim(-1.2, 1.2)
    # Draw an empty line. The comma after 'drawing' is crucial.
    drawing, = plt.plot([],[])
    # Define a function that updates the line data.
    def update(index):
        drawing.set_data(x[:index], y[:index])
        return drawing, # Note the comma!
    def wave(index):
        drawing.set_data(x, np.roll(y, index))
        return drawing,
    a = FuncAnimation(fig, wave, frames=len(x), interval=10)
    plt.show()

def sine_cosine_animation():
    x = np.linspace(0, 2*np.pi, 200)[:-1]
    y1, y2 = np.sin(x), np.cos(x)
    fig = plt.figure()
    plt.xlim(0, 2*np.pi)
    plt.ylim(-1.2, 1.2)
    sin_drawing, = plt.plot([],[])
    cos_drawing, = plt.plot([],[])
    def update(index):
        sin_drawing.set_data(x[:index], y1[:index])
        cos_drawing.set_data(x[:index], y2[:index])
        return sin_drawing, cos_drawing,
    a = FuncAnimation(fig, update, frames=len(x), interval=10)
    plt.show()

def rose_animation_3D():
    theta = np.linspace(0, 2*np.pi, 200)
    x = np.cos(theta) * np.cos(6*theta)
    y = np.sin(theta) * np.cos(6*theta)
    z = theta / 10
    fig = plt.figure()
    ax = fig.gca(projection='3d') # Make the figure 3-D.
    ax.set_xlim3d(-1.2, 1.2) # Use ax instead of plt.
    ax.set_ylim3d(-1.2, 1.2)
    ax.set_aspect("auto")
    drawing, = ax.plot([],[],[]) # Provide 3 empty lists.
    # Update the first 2 dimensions like usual, then update the 3-D component.
    def update(index):
        drawing.set_data(x[:index], y[:index])
        drawing.set_3d_properties(z[:index])
        return drawing,
    a = FuncAnimation(fig, update, frames=len(x), interval=10, repeat=False)
    plt.show()

#sine_animation()
#sine_cosine_animation()
#rose_animation_3D()

def tangent_animation():
    n_pts = 75
    x = np.linspace(-np.pi/2, np.pi/2, n_pts)
    y = np.tan(x)
    fig = plt.figure()

    plt.xlim(-np.pi/2, np.pi/2)
    plt.ylim(-5, 5)

    drawing, = plt.plot([])

    def update(i):
        drawing.set_data(x[i if i < n_pts//2 else n_pts - i : i+n_pts//2 if i < n_pts//2 else 3*n_pts//2 - i], y[i if i < n_pts//2 else n_pts - i : i+n_pts//2 if i < n_pts//2 else 3*n_pts//2 - i])
        return drawing,
    
    a = FuncAnimation(fig, update, frames=len(x), interval=500//n_pts)
    plt.show()

#tangent_animation()
"""