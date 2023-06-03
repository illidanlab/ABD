Anti-Backdoor Data-free Distillation (ABD)
===========

Prepare for running.
1. Install python env.
    ```bash
     # for ZSKT
    conda create --name abd python=3.7
    conda activate abd
    pip install -r requirements.txt  # work with cuda=10.2
     # NOTE if you work with cuda=11.3, do this to update otherwise not working.
     # for cuda=11.3,11.4
     # pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 -U
     # for cuda=11.7
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 -U
    ```