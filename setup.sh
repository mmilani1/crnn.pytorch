# /bin/bash

echo 'Setting up conda env'
conda init bash
conda env create --file requirements.yml

bash -c 'conda activate crnn'
