import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from matplotlib.animation import FuncAnimation

def binarize_pattern(pattern_2d):
    """
    Converts a 2D pattern of 0/1 into a 1D array of +1/-1
    """
    flat = pattern_2d.flatten()
    return np.where(flat == 0, -1, 1)

def train_hopfield(patterns):
    """
    Training a Hopfield network using the Hebbian learning rule:
       W += outer(p, p) for each pattern p,
    with zeroed diagonal (no self-connections).
    """
    n = patterns[0].size
    W = np.zeros((n, n))
    
    for p in patterns:
        W += np.outer(p, p)
    
    np.fill_diagonal(W, 0)  # no self-connections
    return W

def hopfield_recall_asynchronous(W, initial_state, max_updates=4000):
    """
    Asynchronous recall with random neuron updates:
     - Pick one neuron at random, update it, store the state.
     - Repeat up to `max_updates` times.
      
    Returns a list of states so we can animate the step-by-step changes.
    """
    x = deepcopy(initial_state)
    states = [x.copy()]  # store initial state
    
    n = len(x)
    for step in range(max_updates):
        # Pick a random neuron index
        i = np.random.randint(n)
        
        # Compute the weighted sum for neuron i
        raw_value = np.dot(W[i, :], x)
        # Update that single neuron
        x[i] = 1 if raw_value >= 0 else -1
        
        # Store the new state after this single-neuron update
        states.append(x.copy())    
    return states

def to_image(pattern_1d, shape):
    """
    Converting +1/-1 pattern into 0/1 values for imshow, then reshape to `shape`.
    """
    return np.where(pattern_1d == 1, 1, 0).reshape(shape)

def create_circle_pattern(size=11, radius=4, center=None):
    """
    Create a 2D pattern (0/1) in which pixels within `radius` of `center`
    are set to 1 (forming a circle).
    """
    if center is None:
        center = (size // 2, size // 2)
    y0, x0 = center
    
    pattern = np.zeros((size, size), dtype=int)
    for y in range(size):
        for x in range(size):
            if (x - x0)**2 + (y - y0)**2 <= radius**2:
                pattern[y, x] = 1
    return pattern

def create_cross_pattern(size=11, thickness=2):
    """
    Create a 2D pattern (0/1) for a large cross in an `size x size` grid.
    The cross has vertical and horizontal bars with specified `thickness`.
    """
    pattern = np.zeros((size, size), dtype=int)
    mid = size // 2
    
    # Vertical bar
    pattern[:, mid - thickness//2 : mid + (thickness+1)//2] = 1
    # Horizontal bar
    pattern[mid - thickness//2 : mid + (thickness+1)//2, :] = 1
    
    return pattern

def randomize_grid(size=11, density=0.5):
    """
    Generate a random binary grid of given size.
    
    Parameters:
    - size: int, the width and height of the grid (size x size)
    - density: float (0 to 1), probability of a cell being +1 (white).
    
    Returns:
    - A 2D numpy array of shape (size, size) with values +1 or -1.
    """
    grid = np.random.choice([1, -1], size=(size, size), p=[density, 1-density])
    return grid

def main():
    # creating the 11x11 patterns
    size = 11
    pattern_A_2d = create_circle_pattern(size=size, radius=3)
    pattern_B_2d = create_cross_pattern(size=size, thickness=3)
    pattern_C_2d = np.array([[0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0],
                    [0,1,1,1,0,0,0,1,1,1,0],
                    [1,1,1,1,1,0,1,1,1,1,1],
                    [1,1,0,0,1,1,1,0,0,1,1],
                    [1,1,0,0,0,1,0,0,0,1,1],
                    [0,1,1,0,0,0,0,0,1,1,0],
                    [0,0,1,1,0,0,0,1,1,0,0],
                    [0,0,0,1,1,0,1,1,0,0,0],
                    [0,0,0,0,1,1,1,0,0,0,0],
                    [0,0,0,0,0,1,0,0,0,0,0]])
    
    # convert to +1/-1 vectors
    pattern_A = binarize_pattern(pattern_A_2d)
    pattern_B = binarize_pattern(pattern_B_2d)
    pattern_C = binarize_pattern(pattern_C_2d)
    
    # train Hopfield network on these two patterns
    W = train_hopfield([pattern_A, pattern_B, pattern_C])
    
    # generate 'random' start grid
    random_density = np.random.choice([0.2, 0.4, 0.5])
    start_grid = randomize_grid(11, density=random_density).flatten()
    
    # recalling patterns using asynchronous, random updates
    states = hopfield_recall_asynchronous(W, start_grid, max_updates=1000)
    print(f"Collected {len(states)} states (including initial).")
    
    # animate the evolution
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(to_image(states[0], (size, size)), cmap='gray', vmin=0, vmax=1)
    ax.set_title("Hopfield Asynchronous Updates (11x11)")
    ax.axis('off')
    
    def update(frame):
        im.set_data(to_image(states[frame], (size, size)))
        return [im]
    
    ani = FuncAnimation(
        fig,
        update,
        frames=len(states),
        interval=5,   # ms between frames
        blit=True,
        repeat_delay=1000
    )
    
    plt.show()

if __name__ == "__main__":
    main()
