# nuQLOUD

**NUclear-based Quantification of Local Organisation via cellUlar Distributions (nuQLOUD)** is a computational framework designed to generate organizational features from point clouds. It utilizes Voronoi diagrams and kernel density estimations to characterize the arrangement of local point neighborhoods in an object-centered manner.

This repository is part of the publication "In toto analysis of multicellular arrangement reduces embryonic tissue diversity to two archetypes that require specific cadherin expression" by Brambach et al.

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
We recommend creating a fresh environment with Python 3.7.9:

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

Two Jupyter notebooks are included as a usage examples.

1. `nuQLOUD_example_synthetic_data.ipynb`: A lightweight introduction to feature generation with nuQLOUD using generated, synthetic data. A great place to  try out nuQLOUD quickly and on low-performance computers.
2. `nuQLOUD_example_real_data.ipynb`: An example of nuQLOUD feature generation and clustering using real data from a 48 hpf zebrafish embryo. This notebook goes beyond the previous one and also illustrates how organisational motifs and archetypes are identified. Feature generation will be slow on low-performance computers.


## 📂 Repository Overview

    ```
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
