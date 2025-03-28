# Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis
PyTorch implementation of "Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis".

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

## 🚀 Usage

### Training

```bash
bash scripts/run_mvimgnet.sh
```

### Novel View Synthesis

```bash
python spiral.py -s ./mvimgnet --model_path ./exp/mvimgnet/5_views --resolution 512 --near 5 --num_views 5
```

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
