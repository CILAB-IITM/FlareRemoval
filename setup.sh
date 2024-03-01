# setting the flare 7k environment
# cd scripts/Flare7K && python -m venv env && source env/bin/activate && pip install -r requirements.txt && python setup.py develop

# setting the super resolution environment
# mkdir -p scripts/SuperResolution
# cd scripts/SuperResolution && python -m venv env && source env/bin/activate && pip install super_image

python -m venv env
source env/bin/activate && cd scripts/Flare7K && pip install -r requirements.txt && python setup.py develop
source env/bin/activate && pip install super_image