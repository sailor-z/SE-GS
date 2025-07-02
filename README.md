# Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis
PyTorch implementation of "Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis", ICCV 2025.

---

## 📦 Installation

```bash
# Create conda environment
conda create -n gaussian python=3.9
conda activate gaussian

# Install dependencies
bash setup.sh
```

---


## 📊 Evaluations

We evaluate our method on the LLFF, DTU, Mip-NeRF360, and MVImgNet datasets. Note that, due to the stochastic nature of 3DGS, the evaluation results might be slightly different from those reported in the main paper.

### LLFF

#### Data Preparation

1. Download LLFF from **[here](https://drive.google.com/file/d/1kJZuSA188AHSqEk7SOOJjNe3qQt0GUeS/view?usp=sharing)**.

2. Update `base_path` in `tools/colmap_llff.py` to the actual path of your data.

3. Run COLMAP to initialize point clouds and camera parameters:

    ```bash
    python tools/colmap_llff.py
    ```

#### Train and Test

```bash
bash scripts/run_llff.sh {your data path}
```

### Mip-Nerf360

#### Data Preparation

1. Download Mip-Nerf360 from **[here](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip)**.

2. Update `base_path` in `tools/colmap_360.py` to the actual path of your data.

3. Run COLMAP to initialize point clouds and camera parameters:

    ```bash
    python tools/colmap_360.py
    ```

#### Train and Test

```bash
bash scripts/run_360.sh {your data path}
```
---

## ✅ To Do

- ~~Evaluation on LLFF~~
- ~~Evaluation on Mip-Nerf360~~
- Evaluation on DTU
- Evaluation on MvImgNet

---

## 📄 Citation

If you find the project useful, please consider citing:

```
@article{zhao2024self,
  title={Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis},
  author={Zhao, Chen and Wang, Xuan and Zhang, Tong and Javed, Saqib and Salzmann, Mathieu},
  journal={arXiv preprint arXiv:2411.00144},
  year={2024}
}
```
