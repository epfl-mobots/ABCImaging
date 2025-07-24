import cv2 as cv
import numpy as np

def thresholding(img, threshold):
    '''
    Applies a binary threshold to the image.
    Parameters:
    img: The input image as np.uint8.
    threshold: The threshold value (0-255).
    Returns:
    The thresholded image as a binary image.
    '''
    # Apply a threshold to the image
    _, img = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)
    return img

def morph(img, kernel_size = 7, close_first=False,iterations=2):
    # Apply a morphological transformation to the image to remove small islands of noise
    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    if close_first:
        # Apply a closing operation to fill small holes
        img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=iterations)
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=iterations)
    if not close_first:
        # Apply a closing operation to fill small holes
        img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=iterations)
    return img

def remove_small_patches(img, min_size):
    # Find the connected components in the image
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(img, connectivity=8)
    # Remove the small patches
    for i in range(1, num_labels):
        if stats[i, cv.CC_STAT_AREA] < min_size:
            img[labels == i] = 0
    return img

def _region_growing(image, seed_point, visited, mask, threshold:int, max_iterations:int=10000000, min_size:int=700):
    # Get image dimensions
    rows, cols = image.shape[:2]
    # Initialize queue for pixels to visit
    queue = []
    region = []
    queue.append(seed_point)
    iterations = 0
    # Define 4-connectivity neighbors
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while queue :#and iterations < max_iterations:
        iterations += 1
        # Get current pixel from queue
        current_point = queue.pop(0)
        region.append(current_point)
        visited[current_point] = 1

        for neighbor in neighbors:
            # Calculate neighbor coordinates
            x_neighbor, y_neighbor = current_point[0] + neighbor[0], current_point[1] + neighbor[1]

            # Check if neighbor is within image bounds
            if 0 <= x_neighbor < rows and 0 <= y_neighbor < cols:
                # Check if neighbor pixel is not visited
                if visited[x_neighbor, y_neighbor] == 0: #and (rgb2gray(image[x_neighbor, y_neighbor]) < value_threshold):
                    # Calculate gradient descent
                    gradient = abs(int(image[current_point]) - int(image[x_neighbor, y_neighbor]))
                    # Check if gradient is less than threshold
                    if gradient <= threshold:
                        queue.append((x_neighbor, y_neighbor))
                        visited[x_neighbor, y_neighbor] = 1

    if len(region) > min_size: # If the region is big enough, add it to the mask
        for point in region:
            mask[point] = 1

def region_growing(image, gradient_threshold:int=4, value_threshold:int=160, min_size:int=700, verbose:bool=False):

    input_image = np.uint8(image)
    if verbose:
        print(f"Input images type: {type(input_image)}")
        print(f"Input images shape: {input_image.shape}")

    rows, cols = input_image.shape[:2]
    # Initialise visited matrix, taking only the first 2 dimensions of the input image
    visited = np.zeros((input_image.shape[0], input_image.shape[1]))
    mask = np.zeros((input_image.shape[0], input_image.shape[1]))
    # Perform region growing
    for x in range(rows):
        for y in range(cols):
            if (visited[x, y] == 0) and (input_image[x, y] > value_threshold):
                _region_growing(input_image, (x, y), visited, mask, gradient_threshold, value_threshold, min_size=min_size)
    
    return mask