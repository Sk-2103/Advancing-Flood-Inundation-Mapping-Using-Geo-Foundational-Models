This repository contains coode and data source for CVPR 2025 EarthVision submission titled "Advancing Flood Inundation Mapping Using Geo-Foundational Models and Vision Transformers." The original article can be found at.

This article evaluate traditional CNN models, Vision Transformers trained from scratch, and large geo-foundational models (Prithvi and Clay) for flood inundation mapping.

Prithvi Model Implementation
The implementation of the Prithvi model is based on the example provided in the official repository:
https://github.com/NASA-IMPACT/Prithvi-EO-2.0/blob/main/examples/example_landslide4sense.ipynb.

##Training the Prithvi Model
Create a conda environment using:

conda env create -f environment.yaml

1) Define your data directories following the same structure as in the provided scripts.
2) Choose the appropriate band combination and the Prithvi model variant to use.

##Testing and Inference with Prithvi
1) This code uses PyTorch Lightning to directly evaluate the trained model on the test dataset defined in the data loader.
2) Use a Jupyter Notebook to perform inference and save predictions.

#############
Clay Model Implementation
The second geo-foundational model used in this study is Clay, which is open-access.
A detailed description and repository can be found at:
https://github.com/Clay-foundation/model.

Data
This study uses multisensor remote sensing data of 19 flood events globally (FloodPlanet) which can be found at : https://doi.org/10.25739/m69q-8k22

