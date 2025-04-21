# DCDIA  
Dynamic Cascade Disease Information-Aware Drug Combination Recommendation  
![The overall architecture of DCDIA](./figure/Fig1.pdf)

## Overview  
This repository contains code necessary to run DCDIA model.  

## Requirements  
- Python 3.9.21  
- PyTorch 2.5.1  
- CUDA 12.4  
- cuDNN 9.x  

## Datasets  
We provide the dataset in the `data` folder.  

## Documentation  
```bash
.
├── data/                   # Dataset files
│   ├── data_final.pkl
│   ├── ddi_A_final.pkl
│   ├── ddi_mask_H.pkl
│   ├── ehr_adj_final.pkl
│   ├── idx2SMILES.pkl
│   ├── node_features.pkl
│   ├── records_final.pkl
│   ├── substructure_smiles.pkl
│   └── voc_final.pkl
│
├── figure/                 # Model architecture and results
│   ├── Fig1.pdf
│   ├── ...
│   └── Fig9.pdf
│
├── src/                    # Core source code
│   ├── get_node_features.py
│   ├── GNNConv.py
│   ├── layers.py
│   ├── models.py
│   ├── train.py
│   └── util.py
│
└── baseline/               # Baseline implementations
    ├── baseline_near.py
    ├── train_DMNC.py
    ├── train_Leap.py
    ├── train_LR.py
    └── train_Retain.py
```

## Train  
Training script example: `train.py`  

## Test  
The trained model will be automatically stored under the folder `./saved/DCDIA/final.model`.  
Change the default `eval` parameter in `train.py` from `False` to `True`.  

## Authors  
Ziwei Ma @[WinniaMA](https://github.com/WinniaMA)  
Email: <mzw@dlmu.edu.cn>
