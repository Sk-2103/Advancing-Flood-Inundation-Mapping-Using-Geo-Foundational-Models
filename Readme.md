# Advancing Flood Inundation Mapping Using Geo-Foundational Models and Vision Transformers

This repository contains the code and data sources for the CVPR 2025 EarthVision submission titled **"Advancing Flood Inundation Mapping Using Geo-Foundational Models and Vision Transformers."** The original article can be found at *[Provide Link]*.

This study evaluates traditional CNN models, Vision Transformers trained from scratch, and large geo-foundational models (**Prithvi** and **Clay**) for flood inundation mapping.

---

## Prithvi Model Implementation

The implementation of the **Prithvi** model is based on the example provided in the official repository:

➡️ [NASA-IMPACT Prithvi-EO-2.0 Repository](https://github.com/NASA-IMPACT/Prithvi-EO-2.0/blob/main/examples/example_landslide4sense.ipynb)

### Training the Prithvi Model

Create a conda environment using:

```sh
conda env create -f environment.yaml
```

1. Define your data directories following the same structure as in the provided scripts.
2. Choose the appropriate band combination and the Prithvi model variant to use.

### Testing and Inference with Prithvi

1. This code uses **PyTorch Lightning** to directly evaluate the trained model on the test dataset defined in the data loader.
2. Use a Jupyter Notebook to perform inference and save predictions.

---

## Clay Model Implementation

The second geo-foundational model used in this study is **Clay**, which is open-access. 

A detailed description and repository can be found at:

➡️ [Clay Foundation Model Repository](https://github.com/Clay-foundation/model)

### Training and Testing TransNorm, UViT, UNet, and Attention UNet

#### 1. Create a conda environment

```sh
conda env create -f environment_torch.yaml
```

#### 2. Configure Data Paths and Settings

Edit `config/config_setting_other.py` to define:
- **Data paths (`data_path`)** where your datasets are located.
- **Number of channels (`num_channels`)** and other dataset-specific configurations like **image size (`image_size`)** or **batch size (`batch_size`)**.

#### 3. Setup Training Script (`main.ipynb`)

Ensure `main.ipynb` includes the following key steps:
- **Loading and preprocessing data**
- **Initializing and training each model** (TransNorm, UViT, UNet, Attention UNet)
- **Evaluating performance** using appropriate metrics (e.g., accuracy, IoU for segmentation tasks)
- **Saving trained models and results** (optional)

---

## Data

This study uses **multisensor remote sensing data of 19 flood events globally (FloodPlanet)**. The dataset can be accessed at:

➡️ [FloodPlanet Dataset](https://doi.org/10.25739/m69q-8k22)

---

## Citation

If you use this repository, please cite our work:

```
@article{your_citation_here,
  title={Advancing Flood Inundation Mapping Using Geo-Foundational Models and Vision Transformers},
  author={Your Name et al.},
  journal={CVPR EarthVision},
  year={2025}
}
```

---

## License


