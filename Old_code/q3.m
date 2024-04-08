# Creator: Evan Mangat
# Input: Just change path directories to the ones where data is stored
# Output: Visualizations and Graphs

% Load images
img1 = imread('C:\Users\evanm\Documents\Homework\CMPT 340\Project\Blood_Cancer\Blood_Cancer\Sample_2.tiff');
img2 = imread('C:\Users\evanm\Documents\Homework\CMPT 340\Project\Blood_Cancer\Blood_Cancer\Sample_3.tiff');

% Display images side by side, ignoring the alpha channel
figure;
subplot(1,2,1);
imshow(img1(:,:,1:3)); % Display only RGB channels of img1
title('Image 1');

subplot(1,2,2);
imshow(img2(:,:,1:3)); % Display only RGB channels of img2
title('Image 2');


% ----------- 2 
% Assuming img1 is already loaded and is an RGBA image

% Enhance contrast for each of the RGB channels, leaving the alpha channel unchanged
enhancedImg1 = img1; % Initialize with the original image
for k = 1:3 % Loop through the first three channels (RGB)
    enhancedImg1(:,:,k) = imadjust(img1(:,:,k));
end

% Apply noise reduction with median filter to each of the RGB channels, again leaving the alpha channel unchanged
noiseReducedImg1 = enhancedImg1; % Start with the enhanced image
for k = 1:3 % Loop through the RGB channels
    noiseReducedImg1(:,:,k) = medfilt2(enhancedImg1(:,:,k), [3 3]);
end

% Display the original and the processed images, ignoring the alpha channel for display purposes
figure;
subplot(1,2,1);
imshow(img1(:,:,1:3)); % Display only RGB channels of the original image
title('Original Image 1');

subplot(1,2,2);
imshow(noiseReducedImg1(:,:,1:3)); % Display only RGB channels of the processed image
title('Enhanced and Noise Reduced Image 1');


% ----------- 3 
% Display histogram for the first image
figure;
imhist(img1);
title('Histogram of Pixel Intensities for Image 1');

%%
% ----------- 4
% Simple thresholding to create a binary image
thresholdValue = graythresh(img1); % Otsu's method to determine threshold
binaryImage1 = imbinarize(img1, thresholdValue);

% Display the binary image
figure;
imshow(binaryImage1);
title('Binary Image 1 from Thresholding');


% ----------- 5
% Display the image again
figure;
imshow(img1);
title('Image 1 with Annotations');

% Overlaying text annotations
text(50, 50, 'Annotation 1', 'Color', 'yellow', 'FontSize', 12);

%%
% ----------- 6
% Define the directory where your images are stored
imageDir = 'C:\\Users\\evanm\\Documents\\Homework\\CMPT 340\\Project\\Blood_Cancer\\Blood_Cancer\\';

% Initialize arrays to hold mean and standard deviation of intensity values
meanIntensities = zeros(1, 9998); % Adjusted for actual number of images
stdIntensities = zeros(1, 9998); % Adjusted for actual number of images

% Initialize a counter for indexing meanIntensities and stdIntensities
counter = 1;

% Loop through each image file by index, accounting for the jump
for i = [2:999, 1473:9999]
    % Construct the file name according to the naming convention
    fileName = sprintf('Sample_%d.tiff', i);
    filePath = fullfile(imageDir, fileName);
    
    % Check if file exists to handle cases where some numbers might be skipped
    if isfile(filePath)
        % Read the image
        img = imread(filePath);

        % Assuming the images might be multi-plane and selecting the first plane
        if ndims(img) > 2
            img = img(:,:,1);
        end

        % Calculate the mean intensity and store it
        meanIntensities(counter) = mean(img(:)); % Use counter for indexing

        % Calculate the standard deviation of intensity and store it
        stdIntensities(counter) = std(double(img(:))); % Use counter for indexing
    else
        meanIntensities(counter) = NaN; % Use NaN for missing files
        stdIntensities(counter) = NaN; % Similarly for std
    end
    
    % Increment the counter
    counter = counter + 1;
end

% Visualize the mean intensities
figure;
plot(meanIntensities, 'LineWidth', 2);
title('Mean Intensity of Each Image');
xlabel('Image Index (Starting from Sample_2)');
ylabel('Mean Intensity');
xlim([1 9525]);
grid on; % Adding a grid for better visualization

% --- Additional Plots and Analysis ---

% Histogram of Mean Intensities
figure;
histogram(meanIntensities, 'BinWidth', 5);
title('Histogram of Mean Intensities Across All Images');
xlabel('Mean Intensity');
ylabel('Frequency');

% Boxplot of Mean Intensities
figure;
boxplot(meanIntensities, 'Labels', {'Mean Intensities'});
title('Boxplot of Mean Intensities Across All Images');

% Scatter Plot of Mean vs. Standard Deviation of Intensities
figure;
scatter(meanIntensities, stdIntensities);
title('Mean vs. Standard Deviation of Intensities');
xlabel('Mean Intensity');
ylabel('Standard Deviation of Intensity');

% Assuming additional analyses and visualizations follow here based on the earlier discussion


%%
% Define the directory and get a list of TIFF files
imageDir = 'C:\\Users\\evanm\\Documents\\Homework\\CMPT 340\\Project\\Blood_Cancer\\Blood_Cancer\\';
files = dir(fullfile(imageDir, '*.tiff'));

% Initialize an accumulator for the sum of all images
firstImageInfo = imfinfo(fullfile(imageDir, files(1).name)); % Get info of the first image

% Adaptively handle both RGB and RGBA cases. Assuming images could be RGB or RGBA,
% we'll prepare for the maximum expected channels (RGBA = 4 channels).
accumulator = zeros(firstImageInfo.Height, firstImageInfo.Width, 4, 'double');

% Loop through each file and accumulate the sum
numImages = length(files);
for i = 1:numImages
    % Read the image
    img = imread(fullfile(imageDir, files(i).name));
    
    % Ensure we are working with RGBA. If the image does not have an alpha channel,
    % add one with full opacity (255).
    if size(img, 3) == 3
        img = cat(3, img, 255 * ones(size(img, 1), size(img, 2), 'uint8'));
    end
    
    % Convert to double for accumulation
    img = double(img);
    
    % Add the image to the accumulator
    accumulator = accumulator + img;
end

% Calculate the average by dividing by the number of images
averageImage = accumulator / numImages;

% Convert the average image back to uint8
averageImage = uint8(averageImage);

% Display the average image. For displaying, we ignore the alpha channel as it's not needed for visualization.
figure;
imshow(averageImage(:,:,1:3)); % Display only RGB channels
title('Average Image');

% Optionally, save the average image
% Here, the saved image includes the alpha channel.
imwrite(averageImage, fullfile(imageDir, 'averageImage_RGBA.tiff'));


