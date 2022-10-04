# Brain Network Transformer

Brain Network Transformer is the open-source implementation of the NeurIPS 2022 paper [Brain Network Transformer]().


[![Wayfear - BrainNetworkTransformer](https://img.shields.io/static/v1?label=Wayfear&message=BrainNetworkTransformer&color=blue&logo=github)](https://github.com/Wayfear/BrainNetworkTransformer "Go to GitHub repo")
[![stars - BrainNetworkTransformer](https://img.shields.io/github/stars/Wayfear/BrainNetworkTransformer?style=social)](https://github.com/Wayfear/BrainNetworkTransformer)
[![forks - BrainNetworkTransformer](https://img.shields.io/github/forks/Wayfear/BrainNetworkTransformer?style=social)](https://github.com/Wayfear/BrainNetworkTransformer)
![language](https://img.shields.io/github/languages/top/Wayfear/BrainNetworkTransformer?color=lightgrey)
![lines](https://img.shields.io/tokei/lines/github/Wayfear/BrainNetworkTransformer?color=red)
![license](https://img.shields.io/github/license/Wayfear/BrainNetworkTransformer)
![visitor](https://visitor-badge.glitch.me/badge?page_id=BrainNetworkTransformer)
![issue](https://img.shields.io/github/issues/Wayfear/BrainNetworkTransformer)
---


## Usage

```bash
python -m source --multirun datasz=100p model=bnt,fbnetgen,brainnetcnn,transformer dataset=ABIDE,ABCD repeat_time=5 preprocess=mixup
```

- **datasz**, default=(10p, 20p, 30p, 40p, 50p, 60p, 70p, 80p, 90p, 100p)
How much data to use for training. The value is a percentage of the total number of samples in the dataset. For example, 10p means 10% of the total number of samples in the dataset.

- **model**, default=(bnt,fbnetgen,brainnetcnn,transformer)
Which model to use. The value is a list of model names. For example, bnt means Brain Network Transformer, fbnetgen means FBNetGen, brainnetcnn means BrainNetCNN, transformer means VanillaTF.

- **dataset**, default=(ABIDE,ABCD)
Which dataset to use. The value is a list of dataset names. For example, ABIDE means ABIDE, ABCD means ABCD.

- **repeat_time**, default=5. 
How many times to repeat the experiment. The value is an integer. For example, 5 means repeat 5 times.

- **preprocess**, default=(mixup, non_mixup)
Which preprocess to applied. The value is a list of preprocess names. For example, mixup means mixup, non_mixup means the dataset is feeded into models without preprocess.


## Installion

```bash
conda create --name egt python=3.9
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge wandb
conda install -c conda-forge autopep8
pip install hydra-core --upgrade
conda install -c conda-forge scikit-learn
conda install -c conda-forge pandas
```


## Dependencies

  - python=3.9
  - cudatoolkit=11.3
  - torchvision=0.13.1
  - pytorch=1.12.1
  - torchaudio=0.12.1
  - wandb=0.13.1
  - scikit-learn=1.1.1
  - pandas=1.4.3
  - hydra-core=1.2.0


## Citation

```bibtex
@inproceedings{
  kan2022bnt,
  title={BRAIN NETWORK TRANSFORMER},
  author={Xuan Kan and Wei Dai and Hejie Cui and Zilong Zhang and Ying Guo and Carl Yang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022},
}
```