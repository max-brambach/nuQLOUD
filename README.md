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

1. Clone the repository

   ```bash
   git clone https://github.com/max-brambach/nuQLOUD.git
   cd nuQLOUD
   ```
   
2. Create a conda environment (optional but recommended)
We recommend creating a fresh environment with Python 3.7.9::

    ```bash
    conda create -n nuqcloud python=3.7.9
    conda activate nuqcloud
    ```
    
3. Install required Python packages

    ```bash
    pip install -r requirements.txt
    ```
    
4. Install custom voro++

This tool depends on a custom version of voro++. Follow the instructions in that repository to compile and install it.
Make sure the voro++ binary is accessible from your PATH, or adjust the voro_path parameter in your script.

## 🧪 Usage

A Jupyter notebook is included as a usage example:

    ```bash
    jupyter notebook
    ```
Then open and run: nuQLOUD_example_notebook.ipynb

This notebook walks through how to:

Load point cloud data
Generate organizational features
Visualize the resulting structures

## 📂 Repository Overview

    ```bash
    nuQLOUD/
    ├── nuqcloud/                 # Source code
    ├── example_data/             # Example datasets
    ├── nuQLOUD_example_notebook.ipynb
    ├── LICENSE
    ├── requirements.txt
    ├── setup.py
    └── README.md
    ```
   
## 📖 Documentation

Please refer to:

* The example notebook
* Function docstrings in the nuqcloud/ folder
* Issues tab for Q&A


## 🤝 Contributing

We welcome pull requests! Please:

Fork the repo and create a new branch.
Make your changes and write tests if relevant.
Ensure code follows existing formatting.
Submit a pull request with a clear description.


## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.


## 📬 Contact

For questions or feedback, please open an issue on GitHub.
