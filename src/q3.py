from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import numpy as np
import os
from skimage.filters import threshold_otsu

# Creator: Evan Mangat
# Input: Just change path directories to the ones where data is stored
# Output: Visualizations and Graphs

# 1
path_to_img1 = 'C:/Users/evanm/Documents/Homework/CMPT 340/Project/Blood_Cancer/Blood_Cancer/Sample_2.tiff'
path_to_img2 = 'C:/Users/evanm/Documents/Homework/CMPT 340/Project/Blood_Cancer/Blood_Cancer/Sample_3.tiff'

# Load images
img1 = Image.open(path_to_img1)
img2 = Image.open(path_to_img2)

# Display images side by side
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Display the first image, converting to RGB to ignore alpha channel
ax[0].imshow(img1.convert('RGB'))
ax[0].set_title('Image 1')
ax[0].axis('off')  # Hide axes ticks

# Display the second image, also converting to RGB
ax[1].imshow(img2.convert('RGB'))
ax[1].set_title('Image 2')
ax[1].axis('off')  # Hide axes ticks

plt.show()


# ----------------------- 2
# Enhance contrast for each of the RGB channels
enhancer = ImageEnhance.Contrast(img1)
enhanced_img1 = enhancer.enhance(2)  # Adjust the enhancement factor as needed

# Convert the enhanced image to a numpy array for noise reduction
enhanced_img1_np = np.array(enhanced_img1)

# Apply noise reduction with a median filter to each of the RGB channels
# Leaving the alpha channel unchanged (if it exists)
for k in range(3):  # Loop through the RGB channels
    enhanced_img1_np[:,:,k] = median_filter(enhanced_img1_np[:,:,k], size=(3, 3))

# Convert the numpy array back to an image
noise_reduced_img1 = Image.fromarray(enhanced_img1_np)

# Display the original and the processed images, ignoring the alpha channel for display purposes
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Display the original image, converting to RGB to ignore alpha channel
ax[0].imshow(img1.convert('RGB'))
ax[0].set_title('Original Image 1')
ax[0].axis('off')

# Display the processed image
ax[1].imshow(noise_reduced_img1.convert('RGB'))
ax[1].set_title('Enhanced and Noise Reduced Image 1')
ax[1].axis('off')

plt.show()

# ----------------------- 3
# Convert the image into a numpy array
img_np = np.array(img1)

# Calculate the histograms for each RGB channel, ignoring the alpha channel
red_hist = np.histogram(img_np[:, :, 0], bins=256, range=(0, 256))[0]
green_hist = np.histogram(img_np[:, :, 1], bins=256, range=(0, 256))[0]
blue_hist = np.histogram(img_np[:, :, 2], bins=256, range=(0, 256))[0]

# Plotting the histograms
plt.figure(figsize=(10, 4))
plt.plot(red_hist, color='red')
plt.plot(green_hist, color='green')
plt.plot(blue_hist, color='blue')
plt.title('Histogram of Pixel Intensities for Image 1 (Ignoring Alpha Channel)')
plt.xlabel('Pixel value')
plt.ylabel('Frequency')
plt.show()


# --------------------------- 4
# Convert the image to grayscale if it is not already
if img1.mode != 'L':
    img1_gray = img1.convert('L')
else:
    img1_gray = img1

# Convert the grayscale image to a numpy array
img1_np = np.array(img1_gray)

# Use Otsu's method to find the optimal threshold
threshold_value = threshold_otsu(img1_np)

# Apply the threshold to create a binary image
binary_image1 = img1_np > threshold_value

# Display the binary image
plt.figure(figsize=(5, 5))
plt.imshow(binary_image1, cmap='gray')
plt.title('Binary Image 1 from Thresholding')
plt.axis('off')  # Remove axis ticks and labels
plt.show()


# -------------------------- 5
image_dir = 'C:/Users/evanm/Documents/Homework/CMPT 340/Project/Blood_Cancer/Blood_Cancer/'
# Initialize lists to hold mean and standard deviation of intensity values
mean_intensities = []
std_intensities = []

# Adjust your image range according to your specific case
image_range = list(range(2, 1000)) + list(range(1473, 10000))

for i in image_range:
    file_name = f'Sample_{i}.tiff'
    file_path = os.path.join(image_dir, file_name)
    
    # Check if file exists to handle cases where some numbers might be skipped
    if os.path.isfile(file_path):
        # Read the image
        img = Image.open(file_path)
        
        # Convert to grayscale if not already
        img_gray = img.convert('L')
        
        # Convert the image to a numpy array
        img_np = np.array(img_gray)
        
        # Calculate the mean intensity and standard deviation and store them
        mean_intensities.append(np.mean(img_np))
        std_intensities.append(np.std(img_np))
    else:
        mean_intensities.append(np.nan)
        std_intensities.append(np.nan)

# Plotting the results

# Mean Intensity of Each Image
plt.figure()
plt.plot(mean_intensities, linewidth=2)
plt.title('Mean Intensity of Each Image')
plt.xlabel('Image Index (Starting from Sample_2)')
plt.ylabel('Mean Intensity')
plt.xlim([1, len(mean_intensities)])
plt.grid(True)

# Histogram of Mean Intensities
plt.figure()
plt.hist(mean_intensities, bins=np.arange(min(mean_intensities), max(mean_intensities) + 5, 5))
plt.title('Histogram of Mean Intensities Across All Images')
plt.xlabel('Mean Intensity')
plt.ylabel('Frequency')

# Boxplot of Mean Intensities
plt.figure()
plt.boxplot(mean_intensities, labels=['Mean Intensities'])
plt.title('Boxplot of Mean Intensities Across All Images')

# Scatter Plot of Mean vs. Standard Deviation of Intensities
plt.figure()
plt.scatter(mean_intensities, std_intensities)
plt.title('Mean vs. Standard Deviation of Intensities')
plt.xlabel('Mean Intensity')
plt.ylabel('Standard Deviation of Intensity')

plt.show()


# ---------------------- 6
# Define the directory where your images are stored
image_dir = 'C:/Users/evanm/Documents/Homework/CMPT 340/Project/Blood_Cancer/Blood_Cancer/'

# Get a list of TIFF files in the directory
files = [f for f in os.listdir(image_dir) if f.endswith('.tiff')]

# Initialize an accumulator for the sum of all images
# Using the first image to get dimensions and initialize the accumulator
first_image_path = os.path.join(image_dir, files[0])
first_image = Image.open(first_image_path)
first_image_np = np.array(first_image)

# Prepare accumulator with an extra channel for alpha if needed
if first_image_np.shape[2] == 3:  # Check if the image is RGB
    accumulator = np.zeros((first_image_np.shape[0], first_image_np.shape[1], 4), dtype=np.float64)
    accumulator[:, :, :3] = first_image_np  # Copy the first image into the accumulator
    accumulator[:, :, 3] = 255  # Set full opacity for alpha channel
else:  # Assume the image is already RGBA
    accumulator = np.zeros_like(first_image_np, dtype=np.float64)
    accumulator += first_image_np

# Loop through each file and accumulate the sum
for file_name in files[1:]:  # Start from the second image
    img_path = os.path.join(image_dir, file_name)
    img = Image.open(img_path)
    img_np = np.array(img)
    
    if img_np.shape[2] == 3:  # If the image is RGB, add an alpha channel
        alpha_channel = 255 * np.ones((img_np.shape[0], img_np.shape[1], 1), dtype=np.uint8)
        img_np = np.concatenate((img_np, alpha_channel), axis=2)
    
    accumulator += img_np

# Calculate the average by dividing by the number of images
average_image = accumulator / len(files)

# Convert the average image back to uint8
average_image_uint8 = np.uint8(average_image)

# Display the average image, ignoring the alpha channel for visualization
plt.imshow(average_image_uint8[:, :, :3])
plt.title('Average Image')
plt.axis('off')  # Hide axis ticks
plt.show()

# Optionally, save the average image, including the alpha channel
average_image_path = os.path.join(image_dir, 'averageImage_RGBA.tiff')
Image.fromarray(average_image_uint8).save(average_image_path)