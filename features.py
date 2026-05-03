import cv2
import numpy as np

def extract_features(image):
    # Resize to 64x64 to keep the final model lightweight
    image = cv2.resize(image, (64, 64))
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Explicitly configure HOG to fit a 64x64 image
    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hog_features = hog.compute(gray)
    
    # Color Histograms
    hist_b = cv2.calcHist([image], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [32], [0, 256])
    hist_r = cv2.calcHist([image], [2], None, [32], [0, 256])
    
    hist = np.concatenate((hist_b, hist_g, hist_r)).flatten()
    
    # Normalize the color histogram
    if np.sum(hist) > 0:
        hist = hist / np.sum(hist)
    
    # Combine HOG and Color features
    return np.concatenate((hog_features.flatten(), hist))