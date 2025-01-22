# Sentinel-2 Image Matching with LoFTR

This project demonstrates the use of the LoFTR model for keypoint detection and image matching using Sentinel-2 satellite imagery. The project includes dataset preparation, model training, and inference workflows.

## Project Structure

The repository is organized as follows:

- **`data/`**: Directory containing Sentinel-2 images for testing.
  - `image1.jp2`: First satellite image.
  - `image2.jp2`: Second satellite image.
- **`model/`**: Directory storing the model weights.
  - `loftr_weights.pth`: Pretrained weights for the LoFTR model.
- **`notebooks/`**: Contains Jupyter notebooks for step-by-step exploration and visualization.
  - `Sentinel_2_Image_Matching.ipynb`: Demonstrates the process of using LoFTR for image matching.
- **`src/`**: Contains Python scripts for core functionalities.
  - `inference.py`: Script for performing inference with the LoFTR model.
  - `train_model.py`: Script for training the LoFTR model or fine-tuning it on custom datasets.
- **`.gitattributes`**: Git configuration file for handling file types.
- **`README.md`**: Project documentation (this file).
- **`requirements.txt`**: List of dependencies required to run the project.
- **`results/`**: Directory for storing test results and visualizations.
  - `result.jpg`: Output image showing matched keypoints between the input images.
- **`Matching_Recommendation.pdf`**: PDF file containing recommendations for improving image matching performance.


## Steps to Use the Project

### 1. Set up the Environment

To get started, you'll need Python 3.8 or later. First, clone the repository and install the required dependencies.

```bash
# Clone the repository
git clone https://github.com/yourusername/sentinel-image-matching.git
cd sentinel-image-matching

# Install required dependencies
pip install -r requirements.txt
```

### 2. Explore the Notebook
Open the Jupyter notebook in the `notebooks/` directory for an interactive walkthrough of the image matching process.

```bash
jupyter notebook notebooks/Sentinel_2_Image_Matching.ipynb
```

### 3. Perform Inference
Use the `inference.py` script to test the pretrained `LoFTR` model on the provided images. You need to specify the paths to the two images as command-line arguments:

```bash
python src/inference.py --image1 data/image1.jp2 --image2 data/image2.jp2
```
This command will:

- Load the pretrained `LoFTR` model weights from `model/loftr_weights.pth`.
- Match keypoints between the two specified images (`image1.jp2` and `image2.jp2`).
- Show a visualization of the matched keypoints.

If you want to use custom images, replace `data/image1.jp2` and `data/image2.jp2` with the paths to your images.

### 4. Initialize and Test the Model
The `train_model.py` script does not perform training but initializes the classes required for matching keypoints between images. 

Run the script as follows:

```bash
python src/train_model.py
```

This will:

- Initialize the `ImageMatcher` class and load pretrained weights (`loftr_weights.pth`) from the `model/` directory.
- Process the images located at `data/image1.jp2` and `data/image2.jp2`.
- Display the keypoint matches between the two images.

### 5. Results

Below is an example of the output generated after performing inference on `image1.jp2` and `image2.jp2`. The result demonstrates the matched keypoints between the two images:

![Result](results/result.png)

### 6. Additional Notes

- **Dependencies**: All required Python packages are listed in `requirements.txt`.
- **Dataset Preparation**: If using custom images, ensure they are preprocessed to the Sentinel-2 format.
- **Results**: Outputs from inference and processing will be saved in the `results/` directory, including visualizations like `result.png`.
