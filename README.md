# FYP Network Architecture Search (NAS) for Facial Expression Recognition 
This project aims to produce and deploy a deep learning model to perform facial
expression classification tasks by implementing NAS. The implemented NAS is
expected to generate a well-performing neural architecture. The generated neural
architecture is also expected to outperform old and manually designed neural
architectures.
<br/><br/>
The configuration in this repository would perform NAS on CIFAR10 dataset.
# Author:
Chung Wai Kei<br/>
Chiu Chi Hang<br/>
Tse Chung Yin<br/>
# Environment:
Python 3.7, tensorflow-gpu 2.0.0, CUDA 10<br/>
# Usage:
Perform pre-search: python preSearch.py (please manually input the result to mainSearch.py)<br/>
Perform genetic algorithm: python mainSearch.py<br/>
