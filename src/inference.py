import os
import argparse
from train_model import test_image_matching_on_images

# Function to parse command line arguments
def parse_args():
    """
    Parse command-line arguments for image paths.

    Returns:
    Namespace: The parsed arguments with image paths.
    """
    parser = argparse.ArgumentParser(description="Image Matching Test")
    parser.add_argument('--image1', type=str, required=True, help="Path to the first image")
    parser.add_argument('--image2', type=str, required=True, help="Path to the second image")
    return parser.parse_args()

# Main function to run the image matching test
def main():
    # Parse command-line arguments
    args = parse_args()

    # Run the image matching function from the train.py script
    test_image_matching_on_images(args.image1, args.image2)

if __name__ == "__main__":
    main()
