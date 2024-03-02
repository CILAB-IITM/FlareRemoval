python3 -m venv env
source env/bin/activate

pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
pip install -r scripts/Flare7K/requirements.txt
cd scripts/Flare7K && python setup.py develop
pip instal matplotlib
pip install opencv-python