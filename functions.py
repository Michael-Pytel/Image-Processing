import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------- GRAYSCALE ---------------------------
def grayscale_luminance(img):
    """Convert RGB to grayscale using luminance method (vectorized)"""
    return np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


def grayscale_lightness(img):
    """Convert RGB to grayscale using lightness method (vectorized)
    
    Calculates grayscale using the formula: (min(R,G,B) + max(R,G,B)) / 2
    """
    img_float = img.astype(float)
    min_values = np.min(img_float, axis=2)
    max_values = np.max(img_float, axis=2)
    img_gray = (min_values + max_values) / 2
    
    return np.clip(img_gray, 0, 255).astype(np.uint8)



def grayscale_average(img):
    """Convert RGB to grayscale using average method (vectorized)"""
    return np.mean(img, axis=2).astype(np.uint8)

# ----------------Brightness---------------------------
def brightness(img, value):
    """Adjust brightness by adding value to all pixels"""
    return np.clip(img.astype(float) + value, 0, 255).astype(np.uint8)


# ----------------------Contrast------------------------


def contrast(img, param):
    """Adjust contrast using the same formula but vectorized"""
    factor = (259 * (param + 255)) / (255 * (259 - param))
    return np.clip(factor * (img.astype(float) - 128) + 128, 0, 255).astype(np.uint8)


# --------------------Negative ---------------------------
def inverse(img):
    """Invert image colors"""
    return 255 - img

# -----------------Binarization --------------------------

def binarize(img, thresh):
    """Binarize an image (works on both grayscale and RGB)"""
    if len(img.shape) == 3:  # RGB image
        gray = grayscale_luminance(img)
    else:  # Already grayscale
        gray = img
    return (gray > thresh).astype(np.uint8) * 255

# For grayscale images 
def plot_image_with_projections(img):
    # Compute projections
    horizontal_projection = np.sum(img, axis=1)  # Sum across rows
    vertical_projection = np.sum(img, axis=0)    # Sum across columns

    # Create a figure with gridspec to control layout
    fig = plt.figure(figsize=(10, 10), facecolor='black')
    # Layout with horizontal projection on right
    grid = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.05, hspace=0.05)
    
    # Main Image
    ax_img = fig.add_subplot(grid[1, 0])
    ax_img.imshow(img, cmap='gray', aspect='auto')
    ax_img.set_xticks([])
    ax_img.set_yticks([])

    # Horizontal Projection (Right side)
    ax_hor = fig.add_subplot(grid[1, 1], sharey=ax_img)
    # Plot the line (from left to right)
    ax_hor.plot(horizontal_projection, np.arange(len(horizontal_projection)), color='white', linewidth=1.5)
    # Fill below the line
    ax_hor.fill_betweenx(np.arange(len(horizontal_projection)), 0, horizontal_projection, color='white', alpha=0.3)
    ax_hor.set_xticks([])
    ax_hor.set_yticks([])
    ax_hor.set_facecolor('#222222')

    # Vertical Projection (Top)
    ax_ver = fig.add_subplot(grid[0, 0], sharex=ax_img)
    # Plot the line
    ax_ver.plot(np.arange(len(vertical_projection)), vertical_projection, color='white', linewidth=1.5)
    # Fill below the line
    ax_ver.fill_between(np.arange(len(vertical_projection)), 0, vertical_projection, color='white', alpha=0.3)
    ax_ver.set_xticks([])
    ax_ver.set_yticks([])
    ax_ver.set_facecolor('#222222')

    plt.show()


# ===================== FILTERS ===================
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


def apply_custom_kernel(img, kernel):
    """
    Apply a custom kernel to an image.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image (grayscale, RGB, or RGBA)
    kernel : numpy.ndarray
        2D kernel to apply (must be 3x3, 5x5, or 7x7)
        
    Returns:
    --------
    numpy.ndarray
        Filtered image with same dimensions and data type as input
    
    Raises:
    -------
    ValueError
        If kernel size is not 3x3, 5x5, or 7x7
    """
    # Check kernel size
    if kernel.shape not in [(3, 3), (5, 5), (7, 7)]:
        raise ValueError("Kernel must be 3x3, 5x5, or 7x7")
    
    # Apply the kernel using the existing function
    return apply_kernel(img, kernel)


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


def gaussian_blur(img, k, sigma):
    kernel = gaussian_kernel(k, sigma=sigma)
    new_img = apply_kernel(img,kernel)

    return new_img


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


# ============================= Edge Detection ==============================
def roberts_cross(img):
    """
    Apply Roberts Cross edge detection to an image.
    
    The implementation follows the formula from the slide:
    |∇f| = |f(x+1,y) - f(x,y)| + |f(x,y+1) - f(x,y)|
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image (grayscale or RGB)
        
    Returns:
    --------
    numpy.ndarray
        Edge map of the image with same dimensions and data type as input
    """
    # Convert to grayscale if the image is in RGB
    if len(img.shape) == 3:
        gray = grayscale_luminance(img)
    else:
        gray = img.copy()
    
    # Store original data type for later conversion
    original_dtype = img.dtype
    
    # Create output array
    height, width = gray.shape
    result = np.zeros((height, width), dtype=float)
    
    # Calculate horizontal differences: |f(x+1,y) - f(x,y)|
    # For all pixels except the last column
    result[:, :-1] += np.abs(gray[:, 1:].astype(float) - gray[:, :-1].astype(float))
    
    # Calculate vertical differences: |f(x,y+1) - f(x,y)|
    # For all pixels except the last row
    result[:-1, :] += np.abs(gray[1:, :].astype(float) - gray[:-1, :].astype(float))
    
    # Normalize to 0-255 range and convert back to original data type
    return np.clip(result, 0, 255).astype(original_dtype)


def prewitt(img, direction=0):
    """
    Apply Prewitt edge detection to an image with specified direction.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image (grayscale or RGB)
    direction : int
        Direction for edge detection (0-7):
        0: North (top edge)
        1: North-East (top-right edge)
        2: East (right edge)
        3: South-East (bottom-right edge)
        4: South (bottom edge)
        5: South-West (bottom-left edge)
        6: West (left edge)
        7: North-West (top-left edge)
        
    Returns:
    --------
    numpy.ndarray
        Edge map of the image
    """
    # Convert to grayscale if the image is in RGB
    if len(img.shape) == 3:
        gray = grayscale_luminance(img)
    else:
        gray = img.copy()
    
    # Define all 8 Prewitt kernels (as shown in the slide)
    # Each kernel detects edges in a specific direction
    kernels = [
        # 0: North (top edge)
        np.array([
            [ 1,  1,  1],
            [ 0,  0,  0],
            [-1, -1, -1]
        ]),
        
        # 1: North-East (top-right edge)
        np.array([
            [ 0,  1,  1],
            [-1,  0,  1],
            [-1, -1,  0]
        ]),
        
        # 2: East (right edge)
        np.array([
            [-1,  0,  1],
            [-1,  0,  1],
            [-1,  0,  1]
        ]),
        
        # 3: South-East (bottom-right edge)
        np.array([
            [-1, -1,  0],
            [-1,  0,  1],
            [ 0,  1,  1]
        ]),
        
        # 4: South (bottom edge)
        np.array([
            [-1, -1, -1],
            [ 0,  0,  0],
            [ 1,  1,  1]
        ]),
        
        # 5: South-West (bottom-left edge)
        np.array([
            [ 0, -1, -1],
            [ 1,  0, -1],
            [ 1,  1,  0]
        ]),
        
        # 6: West (left edge)
        np.array([
            [ 1,  0, -1],
            [ 1,  0, -1],
            [ 1,  0, -1]
        ]),
        
        # 7: North-West (top-left edge)
        np.array([
            [ 1,  1,  0],
            [ 1,  0, -1],
            [ 0, -1, -1]
        ])
    ]
    
    # Validate direction parameter
    if direction < 0 or direction > 7:
        raise ValueError("Direction must be between 0 and 7")
    
    # Apply the selected kernel using the apply_kernel function
    # This will handle padding and edge cases
    result = apply_kernel(gray, kernels[direction])
    
    return result


def prewitt_all_directions(img):
    """
    Apply Prewitt edge detection in all 8 directions and return the maximum response.
    
    This function follows the slide's guidance that "the gradient direction is determined
    by the mask with the largest values".
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image (grayscale or RGB)
        
    Returns:
    --------
    tuple:
        - magnitude: numpy.ndarray - The edge magnitude (maximum response across all directions)
        - direction: numpy.ndarray - The edge direction (0-7, corresponding to the direction with max response)
    """
    # Convert to grayscale if the image is in RGB
    if len(img.shape) == 3:
        gray = grayscale_luminance(img)
    else:
        gray = img.copy()
    
    # Initialize arrays to store results
    height, width = gray.shape
    magnitude = np.zeros((height, width), dtype=np.float32)
    direction_map = np.zeros((height, width), dtype=np.uint8)
    
    # Apply Prewitt operator in all 8 directions
    responses = []
    for i in range(8):
        responses.append(prewitt(gray, i))
    
    # Convert responses to a single array for easier processing
    responses_array = np.stack(responses, axis=0)
    
    # Get the maximum response and its direction for each pixel
    magnitude = np.max(responses_array, axis=0)
    direction_map = np.argmax(responses_array, axis=0)
    
    return magnitude, direction_map


def prewitt_gradient_magnitude(img):
    """
    Apply Prewitt edge detection using horizontal and vertical kernels 
    and compute the gradient magnitude.
    
    This is a simplified version that only uses the North and East kernels
    (vertical and horizontal edges) and computes the gradient magnitude.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image (grayscale or RGB)
        
    Returns:
    --------
    numpy.ndarray:
        The gradient magnitude
    """
    # Get responses from horizontal and vertical Prewitt operators
    horizontal = prewitt(img, 2)  # East direction
    vertical = prewitt(img, 0)    # North direction
    
    # Compute gradient magnitude
    magnitude = np.sqrt(horizontal.astype(float)**2 + vertical.astype(float)**2)
    
    # Normalize to 0-255 range
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    
    return magnitude

def sobel(img, direction=0):
    """
    Apply Sobel edge detection to an image with specified direction.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image (grayscale or RGB)
    direction : int or float
        Direction for edge detection in degrees.
        Supported angles: 0, 45, 90, 135, 180, 225, 270, 315 degrees
        
    Returns:
    --------
    numpy.ndarray
        Edge map of the image
    """
    # Convert to grayscale if the image is in RGB
    if len(img.shape) == 3:
        gray = grayscale_luminance(img)
    else:
        gray = img.copy()
    
    # Define all 8 Sobel kernels (as shown in the slide)
    # Each kernel detects edges in a specific direction
    kernels = {
        # 0 degrees
        0: np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]),
        
        # 45 degrees
        45: np.array([
            [0, 1, 2],
            [-1, 0, 1],
            [-2, -1, 0]
        ]),
        
        # 90 degrees
        90: np.array([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ]),
        
        # 135 degrees
        135: np.array([
            [2, 1, 0],
            [1, 0, -1],
            [0, -1, -2]
        ]),
        
        # 180 degrees
        180: np.array([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ]),
        
        # 225 degrees
        225: np.array([
            [0, -1, -2],
            [1, 0, -1],
            [2, 1, 0]
        ]),
        
        # 270 degrees
        270: np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]),
        
        # 315 degrees
        315: np.array([
            [-2, -1, 0],
            [-1, 0, 1],
            [0, 1, 2]
        ])
    }
    
    # Validate direction parameter
    if direction not in kernels:
        valid_directions = sorted(kernels.keys())
        raise ValueError(f"Direction must be one of {valid_directions} degrees")
    
    # Apply the selected kernel using the apply_kernel function
    result = apply_kernel(gray, kernels[direction])
    
    return result

def sobel_gradient_magnitude(img):
    """
    Apply Sobel edge detection using horizontal and vertical kernels (0° and 90°)
    and compute the gradient magnitude.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image (grayscale or RGB)
        
    Returns:
    --------
    tuple:
        - magnitude: numpy.ndarray - The gradient magnitude
        - direction: numpy.ndarray - The gradient direction in radians
    """
    # Get responses from horizontal and vertical Sobel operators
    gx = sobel(img, 0)  # 0 degrees - horizontal gradient
    gy = sobel(img, 90) # 90 degrees - vertical gradient
    
    # Convert to float to avoid overflow
    gx = gx.astype(float)
    gy = gy.astype(float)
    
    # Compute gradient magnitude
    magnitude = np.sqrt(gx**2 + gy**2)
    
    # Compute gradient direction
    # Note: atan2 returns angles in radians in range [-π, π]
    direction = np.arctan2(gy, gx)
    
    # Normalize magnitude to 0-255 range
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    
    return magnitude, direction


# =============== MEDIAN FILTER =======================================
def median_filter(img, k=3):
    """
    Apply a median filter to an image.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image (grayscale or RGB)
    k : int
        Size of the filter kernel (must be odd)
        
    Returns:
    --------
    numpy.ndarray
        Filtered image with same dimensions and data type as input
    """
    if k % 2 == 0:
        raise ValueError("k must be an odd number")
    
    # Store original data type and shape for later conversion
    original_dtype = img.dtype
    original_shape = img.shape
    
    # Add padding to handle edge pixels
    kernel_radius = k // 2
    padded_img = reflect_pad(img, kernel_radius)
    
    # Initialize output array based on input dimensions
    if len(original_shape) == 2:  # Grayscale image
        height, width = original_shape
        result = np.zeros((height, width), dtype=original_dtype)
        
        # Apply median filter to grayscale image
        for i in range(height):
            for j in range(width):
                # Extract neighborhood for current pixel
                neighborhood = padded_img[i:i+k, j:j+k]
                # Calculate median
                result[i, j] = np.median(neighborhood)
                
    else:  # RGB or RGBA image
        height, width, channels = original_shape
        result = np.zeros(original_shape, dtype=original_dtype)
        
        # Apply median filter to each channel separately
        for c in range(channels):
            for i in range(height):
                for j in range(width):
                    # Extract neighborhood for current pixel and channel
                    neighborhood = padded_img[i:i+k, j:j+k, c]
                    # Calculate median
                    result[i, j, c] = np.median(neighborhood)
    
    return result