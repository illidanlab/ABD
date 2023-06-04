# Experiments

The code is based on [ZSKT](https://github.com/polo5/ZeroShotKnowledgeTransfer).

## Run Experiments

General
* Single experiment. Distill from a `badnet_grid`-backoored teacher with arch WRN-16-2 to student with arch WRN-16-1.
    ```shell
    export CUDA_VISIBLE_DEVICES=0  # specific GPU
    python main.py --dataset=CIFAR10 --teacher_architecture=WRN-16-2 --student_architecture=WRN-16-1 --trigger_pattern=badnet_grid \
        --sup_backdoor=False --shuf_teacher=0.1 --seeds=3
    ```
* Run sweeps. Find a sweep command at [LOG.md](LOG.md) where you can find all hparams. For example,
    ```shell
    wandb sweep sweeps/cifar10_wrn_poi_sbd.yml
    # get the `wandb agent <some code>` from the CLI output.
    wandb agent <received code>  # this will run one pair of hyper-params from `cifar10_wrn_poi_sbd.yml`.
    ```
    `wandb agent <>` can be run in parallel in different processes, which will auto select different params in `yml` file.


## Distill from poisoned teachers

Evaluate different backdoors with ZSKT.
```sh
wandb sweep sweeps/cifar10_wrn_poi.yml
wandb sweep sweeps/gtsrb_wrn_poi.yml
```

Distill using clean data
```shell
# single run
python kd_distill.py --trigger_pattern=badnet_grid --no_log
wandb sweep sweeps/cifar10_wrn_poi_distill.yml
wandb sweep sweeps/gtsrb_wrn_poi_distill.yml
```

## Defense

* CIFAR10
```shell
wandb sweep sweeps/cifar10_wrn_poi_defense.yml
```

* GTSRB
```shell
wandb sweep sweeps/gtsrb_wrn_poi_defense.yml
```
