# Cloud-API

## Conda setup
```
sudo apt update
sudo apt upgrade
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
bash Miniconda3-py38_4.10.3-Linux-x86_64.sh
```

### Create conda environment
```
conda create --name my_env python=3.8
```

### Activate conda environment
```
conda activate my_env
```

### PyTorch setup
```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

### Requirements setup
```
pip install -r requirements.txt
```

### Deploy API
```
nohup uvicorn main:app --reload --host 0.0.0.0 &
```
