# Install required libraries
import os
import rasterio
import cv2
import matplotlib.pyplot as plt
from rasterio.plot import reshape_as_image
import numpy as np
import torch
import kornia as K
from kornia_moons.viz import draw_LAF_matches
import kornia_moons

# Function to read raster images
def read_raster_image(path):
    """
    Reads a raster image and its metadata.

    Parameters:
    path (str): File path to the raster image.

    Returns:
    tuple: Processed image (H, W, C) and metadata.
    """
    with rasterio.open(path, "r", driver='JP2OpenJPEG') as src:
        raster_image = src.read()
        raster_meta = src.meta
    raster_image = reshape_as_image(raster_image)
    return raster_image, raster_meta

# Visualize images function
def visualize_images(images, scale_factor=0.1):
    """
    Visualize a list of images in a grid.

    Parameters:
    images (list): List of images to display.
    scale_factor (float): Factor by which to scale images for display.
    """
    # Downscale images for visualization
    scaled_images = [
        cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        for image in images
    ]
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(18, 18))
    ax = ax.flatten()

    for i in range(len(scaled_images)):
        ax[i].imshow(scaled_images[i])
        ax[i].axis('off')

    plt.show()

# Image Matcher class
class ImageMatcher:
    """
    A class for matching two images using a simplified keypoint detection algorithm.
    It detects keypoints and matches them between two input images.
    """
    def __init__(self, target_size=None, processing_device=None):
        """
        Initialize the Image Matcher.

        Parameters:
        target_size (tuple): Desired input image size (optional).
        processing_device (str): Device to run the model on (optional).
        """
        self.target_size = target_size
        self.processing_device = processing_device if processing_device else K.utils.get_cuda_device_if_available()
        weights_path = os.path.join(os.path.join(os.path.dirname(__file__)), '..', 'model', 'loftr_weights.pth')
        self.matcher_model = K.feature.LoFTR(pretrained=None).to(self.processing_device).eval()
        if weights_path:
            checkpoint = torch.load(weights_path, map_location=self.processing_device)
            self.matcher_model.load_state_dict(checkpoint)
            print(f"Weights loaded from {weights_path}")
        else:
            print("No weights loaded, using pretrained model")


    def find_matching_keypoints(self, img1, img2, min_confidence=0.8):
        """
        Match keypoints between two images and filter by confidence score.

        Parameters:
        img1 (ndarray): First input image.
        img2 (ndarray): Second input image.
        min_confidence (float): Minimum confidence threshold for keypoint matches.

        Returns:
        dict: Contains the images, keypoints, and confidence scores.
        """
        img1_tensor = self._prepare_image(img1)
        img2_tensor = self._prepare_image(img2)

        input_data = {
            'image0': K.color.rgb_to_grayscale(img1_tensor),
            'image1': K.color.rgb_to_grayscale(img2_tensor)
        }

        with torch.inference_mode():
            correspondences = self.matcher_model(input_data)
        mask = correspondences['confidence'] > min_confidence
        indices = torch.nonzero(mask, as_tuple=True)
        keypoints1 = correspondences['keypoints0'][indices].cpu().numpy()
        keypoints2 = correspondences['keypoints1'][indices].cpu().numpy()
        confidence_scores = correspondences['confidence'][indices].cpu().numpy()

        return {
            'image0': img1_tensor,
            'image1': img2_tensor,
            'keypoints0': keypoints1,
            'keypoints1': keypoints2,
            'confidence': confidence_scores
        }

    def visualize_matches(self, match_info):
        """
        Visualize the matched keypoints between two images.

        Parameters:
        match_info (dict): Contains matched keypoints and images.

        Returns:
        output_fig: The visualization showing the matches.
        """
        inliers = match_info.get('inliers', None)
        output_fig = draw_LAF_matches(
            K.feature.laf_from_center_scale_ori(
                torch.from_numpy(match_info['keypoints0']).view(1, -1, 2),
                torch.ones(match_info['keypoints0'].shape[0]).view(1, -1, 1, 1),
                torch.ones(match_info['keypoints0'].shape[0]).view(1, -1, 1),
            ),
            K.feature.laf_from_center_scale_ori(
                torch.from_numpy(match_info['keypoints1']).view(1, -1, 2),
                torch.ones(match_info['keypoints1'].shape[0]).view(1, -1, 1, 1),
                torch.ones(match_info['keypoints1'].shape[0]).view(1, -1, 1),
            ),
            torch.arange(match_info['keypoints0'].shape[0]).view(-1, 1).repeat(1, 2),
            K.tensor_to_image(match_info['image0']),
            K.tensor_to_image(match_info['image1']),
            inliers,
            draw_dict={
                'inlier_color': (0.1, 1, 0.1),
                'tentative_color': (1, 0.1, 0.1),
                'feature_color': (0.1, 0.1, 1),
                'vertical': False
            }
        )

        return output_fig

    def _prepare_image(self, image):
        """
        Convert image to tensor and resize if necessary.

        Parameters:
        image (ndarray or tensor): Input image to convert.

        Returns:
        Tensor: The processed image tensor.
        """
        MAX_SIZE = 1280
        image_tensor = K.utils.image_to_tensor(image)
        image_tensor = image_tensor.float().unsqueeze(dim=0).to(self.processing_device) / 255.0

        if self.target_size:
            image_tensor = K.geometry.resize(image_tensor, self.target_size, interpolation='area')
        elif max(image_tensor.shape[-1], image_tensor.shape[-2]) > MAX_SIZE:
            image_tensor = K.geometry.resize(image_tensor, MAX_SIZE, side='long', interpolation='area')

        return image_tensor

# Main function for image matching
def test_image_matching_on_images(img1_path, img2_path, target_size=(640, 640)):
    """
    Tests image matching on two images selected from the provided indices and displays the keypoint matches.

    Parameters:
    img1_path (str): Path to image 1.
    img2_path (str): Path to image 2.
    target_size (tuple): Target size for image inputs (default is 640x640).
    """

    # Read the images
    img1, _ = read_raster_image(img1_path)
    img2, _ = read_raster_image(img2_path)

    # Initialize image matcher and find matching keypoints
    image_matcher = ImageMatcher(target_size=target_size)
    matching_results = image_matcher.find_matching_keypoints(img1, img2)

    # Display matched keypoints
    image_matcher.visualize_matches(matching_results)
    plt.show()

# Image paths (relative paths from the train.py file to the data folder)
image1_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'image1.jp2')
image2_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'image2.jp2')

# Run the test
test_image_matching_on_images(image1_path, image2_path)