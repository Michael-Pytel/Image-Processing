import numpy as np
import matplotlib.pyplot as plt

def plot_images(imgs, titles):
    num_images = len(imgs)
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))
    
    # Create a figure with subplots
    fig, axs = plt.subplots(rows, cols, figsize=(8, 8))
    
    # Handle case where there's only one image (axs becomes a single Axes object)
    if num_images == 1:
        axs = np.array([axs])  # Wrap in array to keep indexing consistent
    
    # Flatten axs array for easier 1D indexing if it's 2D
    axs = axs.flatten()
    
    for i in range(num_images):
        # Ensure the image is in a valid format for imshow (e.g., float32, uint8)
        img = imgs[i]
        
        
        if  len(imgs[1].shape) == 2:
            axs[i].imshow(img, cmap='gray')
        else:
            axs[i].imshow(img)
        
        axs[i].set_title(titles[i])
        axs[i].axis('off')
        axs[i].grid(False)
    
    # Turn off unused subplots
    for j in range(num_images, len(axs)):
        axs[j].axis('off')
    
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()