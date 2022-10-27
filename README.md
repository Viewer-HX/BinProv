## BinProv
BinProv is a learning-based compilation provenance identification tool. It will learn context semantics of the machine code using the embedding model and identify the provenance based on the emebedding vector. Our work has been published on [RAID 2022](https://raid2022.cs.ucy.ac.cy/open-access.html).

## Installation
### Packages
- scikit-learn
- numpy
- scipy
- pytorch

We will use `conda` to setup a virtual python environment and install the required packages.
```
conda create -n binprov python=3.7 numpy scipy scikit-learn
conda activate binprov
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```
Note: The version of pytorch must be compatible with the version of Cuda used in your device. Here we recommend:
- PyTorch version >= 1.10.0 
- Python version >= 3.8
- For training new models, you'll also need an NVIDIA GPU and NCCL

### NLP framework
we use Roberta model in BinProv. Here we build the model using [`fairseq`](https://github.com/facebookresearch/fairseq) tool relased by Facebook Research. 


## Citation
```
@inproceedings{xu2022binprov,
author = {He, Xu and Wang, Shu and Xing, Yunlong and Feng, Pengbin and Wang, Haining and Li, Qi and Chen, Songqing and Sun, Kun},
title = {BinProv: Binary Code Provenance Identification without Disassembly},
year = {2022},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
booktitle = {Proceedings of the 25th International Symposium on Research in Attacks, Intrusions and Defenses},
pages = {350–363},
numpages = {14},
location = {Limassol, Cyprus},
series = {RAID '22}
}
```
