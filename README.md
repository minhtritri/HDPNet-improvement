
## Usage

> The training and testing experiments are conducted using PyTorch with an NVIDIA A100-SXM of 40 GB Memory.

### 1. Prerequisites

> Note that HDPNet is only tested on Ubuntu OS with the following environments.

- Creating a virtual environment in terminal: `conda create -n HDPNet python=3.8`.
- Installing necessary packages: `pip install -r requirements.txt`

### 2. Downloading Training and Testing Datasets

- Download the [training set](https://drive.google.com/drive/folders/1V1z3WxDOqZeo8adIu6vmh8rZPf31M_To?usp=drive_link) (COD-TrainDataset) used for training 
- Download the [testing sets](https://drive.google.com/drive/folders/1UndxEPkZFxZwbrVvctb8ZvRpThwB7AND?usp=drive_link) (COD10K-test + CAMO-test + CHAMELEON + NC4K ) used for testing

### 3. Training Configuration

- The pretrained model is stored in [Google Drive](https://drive.google.com/file/d/1fJpCAKDIISC5yQcr4XASalv95hdI5cB4/view?usp=drive_link) . After downloading, please change the file path in the corresponding code.
- Run `train.sh` to train.

### 4. Testing Configuration

Our well-trained model is stored in [Google Drive](https://drive.google.com/file/d/1LfKhIV0cXl_lNpkrLsv4TMvcRMx_IIYW/view?usp=drive_link) After downloading, please change the file path in the corresponding code.

