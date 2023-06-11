The code is based on [CMI](https://github.com/zju-vipa/CMI) and OoD distillation is based on [single-img-extrapolating](https://github.com/yukimasano/single-img-extrapolating).


## CMI

* Attack
```shell
wandb sweep sweeps/cmi_cifar10_poi.yml
```
* Defense
```shell
wandb sweep sweeps/cmi_cifar10.yml
```


### Customization

**Add dataset**:
Edit `get_dataset` and `NORMALIZE_DICT` in [cmi/registry.py](cmi/registry.py).

**Add model**: Edit `MODEL_DICT` in [cmi/registry.py](cmi/registry.py) to add model architecture.
To set pre-trained model files, edit `get_pretrained_path` in [cmi/](cmi/utils/config.py).


## OoD

* Attack
```shell
wandb sweep sweeps/ood_cifar10_poi.yml
```
* Defense
```shell
wandb sweep sweeps/ood_cifar10.yml
```
