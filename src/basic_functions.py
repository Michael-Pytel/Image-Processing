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