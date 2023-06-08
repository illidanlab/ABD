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


<table style="text-align: center">
    <tr>
        <td><b>Trigger</b></td>
        <td><b>Teacher</b></td>
        <td colspan="3"><b>Student Acc/ASR</b></td>
    </tr>
    <tr>
        <td></td>
        <td>Acc/ASR</td>
        <td>ZSKT</td>
        <td>ZSKT+ABD</td>
        <td>Clean KD</td>
    </tr>
    <tr>
        <td>BadNets (grid)</td>
        <td>92.1/99.9</td>
        <td>71.9/96.9</td>
        <td>68.3/0.7</td>
        <td>74.6/4.3</td>
    </tr>
    <tr>
        <td>Trojan WM</td>
        <td>93.8/100</td>
        <td>82.7/93.9</td>
        <td>78.2/22.5</td>
        <td>77.5/11.1</td>
    </tr>
    <tr>
        <td>Trojan 3x3</td>
        <td>93.4/98.7</td>
        <td>80.9/96.8</td>
        <td>71.7/33.3</td>
        <td>72.9/1.7</td>
    </tr>
    <tr>
        <td>Blend</td>
        <td>93.9/99.7</td>
        <td>77.0/74.4</td>
        <td>71.5/23.1</td>
        <td>78.0/4.3</td>
    </tr>
    <tr>
        <td>Trojan 8x8</td>
        <td>93.7/99.6</td>
        <td>80.5/57.2</td>
        <td>72.6/17.8</td>
        <td>75.2/9.3</td>
    </tr>
    <tr>
        <td>BadNets (sq)</td>
        <td>93.4/97.8</td>
        <td>80.8/37.8</td>
        <td>77.9/1.9</td>
        <td>76.2/9.1</td>
    </tr>
    <tr>
        <td>CL</td>
        <td>91.2/94.3</td>
        <td>76.8/17.5</td>
        <td>67.4/10.2</td>
        <td>69.4/2.1</td>
    </tr>
    <tr>
        <td>Sig</td>
        <td>90.5/97.3</td>
        <td>77.9/0.0</td>
        <td>72.2/0.</td>
        <td>77.4/0.</td>
    </tr>
    <tr>
        <td>l2_inv</td>
        <td>93.9/100</td>
        <td>82.0/0.3</td>
        <td>70.7/1.9</td>
        <td>77.2/1.2</td>
    </tr>
    <tr>
        <td>l0_inv</td>
        <td>92.4/99.6</td>
        <td>72.8/8.3</td>
        <td>69.4/0.</td>
        <td>79.2/3.7</td>
    </tr>
</table>


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
