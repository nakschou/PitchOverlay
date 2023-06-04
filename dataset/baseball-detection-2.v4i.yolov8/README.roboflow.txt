
baseball-detection-2 - v4 2023-06-02 3:09pm
==============================

This dataset was exported via roboflow.com on June 2, 2023 at 10:11 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 4153 images.
Baseballs are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)
* Auto-contrast via adaptive equalization

The following augmentation was applied to create 2 versions of each source image:
* Randomly crop between 0 and 15 percent of the image
* Random brigthness adjustment of between -25 and +25 percent
* Random exposure adjustment of between -9 and +9 percent
* Salt and pepper noise was applied to 5 percent of pixels


