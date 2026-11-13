# Dataset setup

Do not commit raw radiographs or dataset archives to GitHub.

The Kaggle notebook expects:

- GRAZPEDWRI-DX at `/kaggle/input/datasets/harigopig/fracture`
- FracAtlas at `/kaggle/input/datasets/harigopig/atlasf/FracAtlas`

Dataset Download Links

HeatAgent uses the following publicly available research datasets:

1. GRAZPEDWRI-DX: "Download from Figshare" `https://figshare.com/articles/dataset/GRAZPEDWRI-DX/14825193`
2. FracAtlas: "Download from Figshare" `https://figshare.com/articles/dataset/The_dataset/22363012`

Download both datasets and extract them into the "data/raw/" directory before running the project locally.

Attach these datasets to the Kaggle notebook using Kaggle's dataset input panel. For local runs, download them separately and update the dataset paths in the notebook / manifest-building cells. Keep private or restricted data outside this repository and follow the datasets' access terms.
