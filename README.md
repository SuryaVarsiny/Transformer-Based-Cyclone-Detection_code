 Transformer-Based Cyclone Detection using Spatiotemporal Features

🔹 1. Project Description
This work presents a transformer-based framework for automated tropical cyclone detection from satellite imagery. The proposed model integrates a Data-efficient Image Transformer (DeiT) with physically interpretable spatiotemporal descriptors to enhance classification reliability and transparency.

🔹 2. Dataset Description
The model is trained and evaluated on satellite imagery from the North Indian Ocean (NIO) region.

- Input: Infrared satellite images
- Task: Binary classification
  - Cyclone
  - Non-Cyclone

The dataset captures diverse atmospheric conditions, enabling robust learning of cyclone-specific spatial and temporal patterns.

 🔹 3. Model Architecture and Key Algorithms

 🔸 Transformer-Based Feature Learning
- Patch Embedding (16×16 patches → 196 tokens)
- DeiT encoder with multi-head self-attention
- Global context modeling for cyclone structure detection

 🔸 Spatiotemporal Feature Extraction
The following image-derived metrics are computed:

- Mean Brightness: Represents cloud-top intensity
- Texture Energy: Computed using Sobel gradients to capture structural variation
- Optical Flow (Farneback): Measures temporal cloud motion between consecutive frames

These features provide physical interpretability and complement transformer predictions.

🔸 Decision Mechanism
- Sigmoid output for cyclone probability
- Threshold-based classification:
  - Cyclone ≥ 0.75
  - Non-Cyclone ≤ 0.25
  - Otherwise: Uncertain

🔹 4. Dependencies and Requirements

All required dependencies are listed in the `requirements.txt` file.

 Install using:
```bash
pip install -r requirements.txt
```

🔹 5. Reproducibility Section

This repository contains the official implementation of the manuscript submitted to The Visual Computer. The code is provided to ensure reproducibility of the experimental results. All necessary components, including model architecture, inference pipeline, and feature extraction methods, are included to facilitate independent verification and evaluation.

🔹 6. Citation
If you use this work, please cite:


Paper Title: "Transformer-Based Cyclone Detection: Leveraging Spatiotemporal Features from Satellite Imagery" 

Author Names: P. Karthikeyan, B. Sakthivel, S. S. Surya Varsiny, S. M. Tejesvini.

Journal: The Visual Computer

Year: 2026  

This code is directly associated with the manuscript currently under review at The Visual Computer. Users are encouraged to cite the corresponding paper when using this repository for research or development purposes.

🔹 7. DOI:10.5281/zenodo.19249182

https://doi.org/10.5281/zenodo.19249181

[![DOI](https://zenodo.org/badge/1193375915.svg)](https://doi.org/10.5281/zenodo.19249181)
