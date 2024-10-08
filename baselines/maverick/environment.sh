module purge
module load anaconda/3
module load cuda/12.1.1
conda create -n maverick_env python=3.11 pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda activate maverick_env
pip install maverick-coref
pip install datasets>=2.14
