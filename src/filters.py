import numpy as np
import matplotlib.pyplot as plt





def sharpness(img,strength=1):
    # Center value increases with strength
    center_value = 1 + 4 * strength
    # Edge values get more negative with strength
    edge_value = -1 * strength
    
    kernel = np.array([
        [0, edge_value, 0],
        [edge_value, center_value, edge_value],
        [0, edge_value, 0]
    ])
    
    new_img = apply_kernel(img, kernel)
    

    return new_img



def apply_kernel_no_clip(img, kernel):
    height = len(img)
    width = len(img[0])

    new_img = np.copy(img)
    new_img = new_img.astype(float)

    kernel_radius = int((len(kernel)-1)/2)
    for i in range(kernel_radius, height-kernel_radius):
        for j in range(kernel_radius, width-kernel_radius):
            for z in range(3):

                chunk = img[i-kernel_radius:i+kernel_radius+1, j-kernel_radius:j+kernel_radius+1,z]

                new_img[i][j][z] = np.sum(np.multiply(kernel, chunk))

    return 

def apply_kernel(img, kernel):
    """
    Apply a kernel to an image with proper padding.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image (grayscale, RGB, or RGBA)
    kernel : numpy.ndarray
        2D kernel to apply
        
    Returns:
    --------
    numpy.ndarray
        Filtered image with same dimensions and data type as input
    """
    # Store original data type for later conversion
    original_dtype = img.dtype
    
    # Add padding before applying kernel
    kernel_radius = kernel.shape[0] // 2
    padded_img = reflect_pad(img, kernel_radius)
    
    # Initialize output array
    result = np.zeros_like(padded_img, dtype=float)
    
    # Get number of channels (if any)
    if len(img.shape) == 2:  # Grayscale image
        channels = 1
        # Reshape for easier processing
        padded_img_reshaped = padded_img[:, :, np.newaxis]
        result_reshaped = result[:, :, np.newaxis]
    else:  # Color image
        channels = img.shape[2]  # Works for RGB (3) or RGBA (4) or any number of channels
        padded_img_reshaped = padded_img
        result_reshaped = result
    
    # Apply kernel to each channel
    height, width = padded_img.shape[:2]
    
    # We can vectorize the operation for each channel separately
    for c in range(channels):
        for i in range(kernel_radius, height - kernel_radius):
            for j in range(kernel_radius, width - kernel_radius):
                # Extract the local region
                region = padded_img_reshaped[i-kernel_radius:i+kernel_radius+1, 
                                           j-kernel_radius:j+kernel_radius+1, 
                                           0 if len(img.shape) == 2 else c]
                
                # Apply kernel
                result_reshaped[i, j, 0 if len(img.shape) == 2 else c] = np.sum(region * kernel)
    
    # Remove padding to get back original image size
    if len(img.shape) == 2:
        # Convert back to 2D for grayscale
        result = result_reshaped[:, :, 0]
    
    final_result = remove_padding(result, kernel_radius)
    
    # Convert back to original data type
    return np.clip(final_result,0,255).astype(original_dtype)


def mean_filter(img, k):
    if k%2 ==0:
        raise ValueError("k must be an odd number")
    
    kernel = np.ones((k,k)) / k**2 #element-wise
    
    new_img = apply_kernel(img, kernel)
    return new_img


def gaussian_kernel(size, sigma, size_y=None):
    """
    Create a Gaussian kernel with specified size and standard deviation (sigma).
    
    Parameters:
    -----------
    size : int
        Half the width of the kernel (full width will be 2*size+1)
    sigma : float
        Standard deviation of the Gaussian distribution
    size_y : int, optional
        Half the height of the kernel. If None, uses size for both dimensions
        
    Returns:
    --------
    numpy.ndarray
        Normalized Gaussian kernel
    """
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)
    
    x, y = np.mgrid[-size:size+1, -size_y:size_y+1]
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return g / g.sum()


def reflect_pad(image, pad_width):
    """
    Apply reflect padding to an image (numpy array).
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image as a numpy array (2D for grayscale, 3D for color)
    pad_width : int or tuple
        Size of padding on each border. Can be an int (same padding all around)
        or a tuple specifying padding for each dimension
        
    Returns:
    --------
    numpy.ndarray
        Padded image array
    """
    # Convert to tuple if pad_width is an integer
    if isinstance(pad_width, int):
        if len(image.shape) == 2:  # Grayscale image
            pad_width = ((pad_width, pad_width), (pad_width, pad_width))
        else:  # Color image
            pad_width = ((pad_width, pad_width), (pad_width, pad_width), (0, 0))
    
    # Use numpy's built-in pad function with reflect mode
    padded_image = np.pad(image, pad_width, mode='reflect')
    
    return padded_image

def remove_padding(padded_image, pad_width):
    """
    Remove padding from an image.
    
    Parameters:
    -----------
    padded_image : numpy.ndarray
        Padded image as a numpy array
    pad_width : int or tuple
        Size of padding that was applied. Should match what was used for padding
        
    Returns:
    --------
    numpy.ndarray
        Original image with padding removed
    """
    # Handle different pad_width formats
    if isinstance(pad_width, int):
        if len(padded_image.shape) == 2:  # Grayscale image
            return padded_image[pad_width:-pad_width, pad_width:-pad_width]
        else:  # Color image
            return padded_image[pad_width:-pad_width, pad_width:-pad_width, :]
    
    else:  # If pad_width is a tuple
        # Extract the padding values for each dimension
        if len(padded_image.shape) == 2:  # Grayscale image
            return padded_image[
                pad_width[0][0]:-pad_width[0][1] if pad_width[0][1] > 0 else None,
                pad_width[1][0]:-pad_width[1][1] if pad_width[1][1] > 0 else None
            ]
        else:  # Color image
            return padded_image[
                pad_width[0][0]:-pad_width[0][1] if pad_width[0][1] > 0 else None,
                pad_width[1][0]:-pad_width[1][1] if pad_width[1][1] > 0 else None,
                :
            ]

def gaussian_blur(img, k, sigma):
    kernel = gaussian_kernel(k, sigma=sigma)
    new_img = apply_kernel(img,kernel)

    return new_img






def plot_histogram(img):
    plt.figure(figsize=(10, 6), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('#222222')

    if len(img.shape) == 2:  # Grayscale image
        hist, bins = np.histogram(img, bins=256, range=(0, 256))
        plt.fill_between(bins[:-1], hist, color='white', alpha=0.7)  # White fill for grayscale
        plt.title("Grayscale Histogram", fontsize=14, color='white')
        plt.legend(["Grayscale"], fontsize=12, facecolor='#333333', edgecolor='white', loc="upper right")
    else:  # RGB image
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            hist, bins = np.histogram(img[..., i], bins=256, range=(0, 256))
            plt.fill_between(bins[:-1], hist, color=color, alpha=0.5)
        plt.title("RGB Histogram", fontsize=14, color='white')
        plt.legend(["Red", "Green", "Blue"], fontsize=12, facecolor='#333333', edgecolor='white', loc="upper right")

    plt.xlabel("Pixel Intensity", fontsize=12, color='white')
    plt.ylabel("Frequency", fontsize=12, color='white')
    plt.xlim([0, 255])
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.tick_params(axis='both', colors='white')

    plt.show()
