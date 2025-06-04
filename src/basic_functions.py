import matplotlib.pyplot as plt
import numpy as np



# ------------------------------------- GRAYSCALE ---------------------------
def grayscale_luminance(img):
    height = len(img)
    width = len(img[0]) 
    
    
    img_gray = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            r, g, b = img[i][j]
            img_gray[i][j] = int(0.299*r + 0.587*g + 0.114*b)
    
    return img_gray


def grayscale_lightness(img):
    height = len(img)
    width = len(img[0])

    img = img.astype(float)
    img_gray = np.zeros((height, width), dtype=np.float16)

    for i in range(height):
        for j in range(width):
            r, g, b = img[i][j]
            img_gray[i][j] = (min(r,g,b) + max(r,g,b))/2
            

    img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)
    return img_gray



def grayscale_average(img):
    height = len(img)
    width = len(img[0])
    
    img = img.astype(float)
    img_gray = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            r, g, b = img[i][j]
            img_gray[i,j] = int((r + g + b) / 3)

    img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)
    return img_gray

# ----------------Brightness---------------------------
def brightness(img, value):
    img = img.astype(float)
    img = img + value
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


# ----------------------Contrast------------------------

def contrast(img, param):
    factor = (259 * (param + 255)) / (255 * (259 - param))
    img = img.astype(np.float32)
    img = factor * (img - 128) + 128
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img


# --------------------Negative ---------------------------
def inverse(img):
    img = img.astype(float)
    img = 255 - img 
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


# -----------------Binarization --------------------------

def binarize(img, thresh):
    gray = grayscale_luminance(img)
    bin = (gray > thresh).astype(np.uint8) * 255

    return bin

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