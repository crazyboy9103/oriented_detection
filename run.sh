# Create a virtual environment
python -m venv env

# Activate the virtual environment
source env/bin/activate

# Install PyTorch and torchvision
pip install torch torchvision

# Install PyTorch Lightning
pip install pytorch-lightning

python entrypoint.py --model_type rotated

python entrypoint.py --model_type oriented
