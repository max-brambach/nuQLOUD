# nuQLOUD

**NUclear-based Quantification of Local Organisation via cellUlar Distributions (nuQLOUD)** is a computational framework designed to generate organizational features from point clouds. It utilizes Voronoi diagrams and kernel density estimations to characterize the arrangement of local point neighborhoods in an object-centered manner.

## 🚀 Features

- **Voronoi-Based Analysis:** Computes 3D Voronoi diagrams to assess spatial relationships between points.
- **Kernel Density Estimation:** Estimates local densities to identify structural patterns.
- **Customizable Parameters:** Allows users to adjust analysis parameters to fit specific datasets.

## 📦 Installation

### Prerequisites

- Python 3.7 or higher
- [voro++](https://github.com/max-brambach/voro): A custom version required for Voronoi diagram generation.

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/max-brambach/nuQLOUD.git
   cd nuQLOUD
   ```
   
2. Create a conda environment (optional but recommended):
We recommend creating a fresh environment with Python 3.10::

    ```bash
    conda create -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
