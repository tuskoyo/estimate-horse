# Horse Analysis Tool
## Environment
- Python version: 3.8.18
- torch version: 1.13.1+cu117
- torchvision version: 0.14.1+cu117
- mmpose version: 1.2.0
- cuda version: 11.7
- compiler information: GCC 9.3

## Installation
### 1. Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```
### 2. Install Pytorch
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

### 3. Install MMEngine and MMCV using MIM
```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
```

### 4. Install mmdet
```bash
mim install "mmdet>=3.1.0"
```

### 5. Install mmpose
```bash
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .
cd ..
```

## USAGE


### Example
```bash
python dev_fold.py --input INPUT_PATH --output OUTPUT_PATH
```

Arguments

--input INPUT_PATH: Input video file or directory containing video files.

--output OUTPUT_PATH: Output directory where results will be saved.

Example Usage

Process a directory of test videos and save results:

python dev_fold.py --input ./test-data/ --output ./test-output
