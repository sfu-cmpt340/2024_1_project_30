import cv2
import os
import pandas as pd
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from scipy.stats import skew, kurtosis

# Define your feature extraction function here
def extract_features(image_path):
    # Load the image
    image = io.imread(image_path)
    
    # Convert to grayscale
    gray_image = rgb2gray(image)
    
    # Thresholding to segment the cells from the background
    thresh = threshold_otsu(gray_image)
    binary_image = gray_image > thresh
    
    # Label the image
    labeled_image = label(binary_image)
    regions = regionprops(labeled_image, intensity_image=gray_image)
    
    # Prepare an array to store feature data
    feature_data = []
    
    for region in regions:
        # Skip small regions that could be noise
        if region.area < 50:
            continue
        
        # Morphological features
        area = region.area
        perimeter = region.perimeter
        eccentricity = region.eccentricity
        solidity = region.solidity
        
        # Color features (within the region of the cell)
        minr, minc, maxr, maxc = region.bbox
        cell_region = image[minr:maxr, minc:maxc]
        mean_intensity = np.mean(cell_region, axis=(0, 1))
        std_intensity = np.std(cell_region, axis=(0, 1))
        skew_intensity = skew(cell_region.reshape(-1, 3), axis=0)
        kurt_intensity = kurtosis(cell_region.reshape(-1, 3), axis=0)
        
        # Create a dictionary for the features
        features = {
            'Area': area,
            'Perimeter': perimeter,
            'Eccentricity': eccentricity,
            'Solidity': solidity,
            'Mean Intensity R': mean_intensity[0],
            'Mean Intensity G': mean_intensity[1],
            'Mean Intensity B': mean_intensity[2],
            'Std Intensity R': std_intensity[0],
            'Std Intensity G': std_intensity[1],
            'Std Intensity B': std_intensity[2],
            'Skewness R': skew_intensity[0],
            'Skewness G': skew_intensity[1],
            'Skewness B': skew_intensity[2],
            'Kurtosis R': kurt_intensity[0],
            'Kurtosis G': kurt_intensity[1],
            'Kurtosis B': kurt_intensity[2],
        }
        
        feature_data.append(features)
    
    return feature_data

# Now you start processing the folders and images
# NOTE: CHANGE THIS ACCORDING TO YOUR SPECIFIC SETUP AND LOCATION OF DATABASE.
root_dir = 'C:/Users/evanm/Documents/Homework/CMPT 340/Project_Test' 
cell_types = [
    'ABE', 'ART', 'BAS', 'BLA', 'EBO', 'EOS', 'FGC', 'HAC',
    'KSC', 'LYI', 'LYT', 'MMZ', 'MON', 'MYB', 'NGB', 'NGS',
    'NIF', 'OTH', 'PEB', 'PLM', 'PMO'
]
data = []

# Loop over cell type abbreviations to access each folder
for cell_type in cell_types:
    # Path to the current cell type's folder
    cell_type_path = os.path.join(root_dir, cell_type)
    num_images = 0  # Counter to keep track of the number of images processed

    # os.walk yields a 3-tuple (dirpath, dirnames, filenames)
    for dirpath, dirnames, filenames in os.walk(cell_type_path):
        # Only print when we're in a new directory
        print(f"Now processing directory: {dirpath}")
        for image_filename in filenames:
            if image_filename.lower().endswith('.jpg'):
                # Construct the full path to the image
                image_path = os.path.join(dirpath, image_filename)

                # Extract features from the image
                features = extract_features(image_path)

                # Add the cell type label to your features
                for feature in features:
                    feature['Cell_Type'] = cell_type
                    data.append(feature)
                
                num_images += 1  # Increment the counter

    # After finishing each cell type folder, print the number of images processed
    print(f"Finished processing {num_images} images in folder: {cell_type_path}")



# Convert to a DataFrame
df = pd.DataFrame(data)

# Get a list of all column names except 'Cell_Type'
columns_except_cell_type = [col for col in df.columns if col != 'Cell_Type']

# Reorder the DataFrame to have 'Cell_Type' first, followed by the other columns
df = df[['Cell_Type'] + columns_except_cell_type]

# Now you can write the DataFrame to a CSV file
# NOTE: CHANGE THIS TO YOUR DESIRED FILE PATH
csv_file_path = 'C:/Users/evanm/Documents/Homework/CMPT 340/Project_Test/cell_features.csv'
df.to_csv(csv_file_path, index=False)

print(f"Features exported to CSV file at: {csv_file_path}")
# You can now save the DataFrame to a CSV file or use it for machine learning model training

# Group the DataFrame by 'Cell_Type' and calculate the mean for each group
average_features_per_cell_type = df.groupby('Cell_Type').mean()

# Display the average features for each cell type
print(average_features_per_cell_type)

# If you want to save the averages to a CSV file
# NOTE: CHANGE THIS TO YOUR DESIRED FILE PATH
average_features_csv_path = 'C:/Users/evanm/Documents/Homework/CMPT 340/Project_Test/average_features_per_cell_type.csv'
average_features_per_cell_type.to_csv(average_features_csv_path)

print(f"Averages saved to CSV file at: {average_features_csv_path}")
