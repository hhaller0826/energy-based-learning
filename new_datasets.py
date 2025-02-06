import numpy as np

def make_interlocking_circles(n_samples=1000, noise=0.05, random_state=None):
    """
    Generates a 3D dataset of two interlocking circles.
    
    Parameters:
    - n_samples (int): Total number of points (split evenly between the two circles).
    - noise (float): Standard deviation of Gaussian noise.
    - random_state (int, optional): Random seed for reproducibility.
    
    Returns:
    - X (ndarray): (n_samples, 3) array of 3D points.
    - y (ndarray): (n_samples,) array of labels (0 or 1).
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Ensure even split of samples between the two circles
    n_samples_per_circle = n_samples // 2

    # Generate the first circle in the XY-plane (centered at (0,0,0))
    theta1 = np.linspace(0, 2 * np.pi, n_samples_per_circle)
    x1 = np.cos(theta1)
    y1 = np.sin(theta1)
    z1 = np.zeros_like(theta1)

    # Generate the second circle in the YZ-plane (centered at (0,1,0)), interlocking with the first
    theta2 = np.linspace(0, 2 * np.pi, n_samples_per_circle)
    x2 = np.zeros_like(theta2)
    y2 = np.ones_like(theta2) + np.cos(theta2)  # Shifted up by 1
    z2 = np.sin(theta2)

    # Stack the data
    X = np.vstack([
        np.column_stack([x1, y1, z1]),  # First circle
        np.column_stack([x2, y2, z2])   # Second circle (interlocking)
    ])

    # Labels: 0 for first circle, 1 for second circle
    y = np.hstack([np.zeros(n_samples_per_circle), np.ones(n_samples_per_circle)])

    # Add Gaussian noise
    X += np.random.normal(scale=noise, size=X.shape)

    return X, y