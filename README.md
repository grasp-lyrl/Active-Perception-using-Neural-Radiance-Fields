# Active Perception using Neural Radiance Fields ([Paper](https://arxiv.org/abs/2310.09892))
Authors: Siming He, Christopher D. Hsu∗, Dexter Ong∗, Yifei Simon Shao, Pratik Chaudhari

## Abstract
We study active perception from first principles to argue that an autonomous agent performing active perception should maximize the mutual information that past observations posses about future ones. Doing so requires (a) a representation of the scene that summarizes past observations and the ability to update this representation to incorporate new observations (state estimation and mapping), (b) the ability to synthesize new observations of the scene (a generative model), and (c) the ability to select control trajectories that maximize predictive information (planning). This motivates a neural radiance field (NeRF)-like representation which captures photometric, geometric and semantic properties of the scene grounded. This representation is well-suited to synthesizing new observations from different viewpoints. And thereby, a sampling-based planner can be used to calculate the predictive information from synthetic observations along dynamically-feasible trajectories. We use active perception for exploring cluttered indoor environments and employ a notion of semantic uncertainty to check for the successful completion of an exploration task. We demonstrate these ideas via simulation in realistic 3D indoor environments.


## Video of Active Perception in Habitat Simulation Scene
In each video, the third-person view and top view of active perception are shown on the left. The ground truth and NeRF synthesis of image, depth, and semantic segmentation in first-person view are shown on the right. During each trajectory, the NeRF synthesis result looks bad because the agent is moving to areas with higher predictive information (usually areas with less reconstruction quality). After each trajectory, the NeRF is trained on collected observations and the NeRF synthesis result becomes better.  
### Scene 1


https://github.com/grasp-lyrl/Active-Perception-using-Neural-Radiance-Fields/assets/69362937/ed965c50-9be2-4cf9-9d37-448d2e152865


### Scene 2


https://github.com/grasp-lyrl/Active-Perception-using-Neural-Radiance-Fields/assets/69362937/ef864e3b-68a0-4c4f-8a80-1ba5ed6642eb


### Scene 3


https://github.com/grasp-lyrl/Active-Perception-using-Neural-Radiance-Fields/assets/69362937/d1fce785-59c5-458a-8cc5-255a789043bf


## Setup
### Installation
```
# clone repo
git clone git@github.com:grasp-lyrl/Active-Perception-using-Neural-Radiance-Fields.git

# set up environment
conda create -n anmap python=3.9 cmake=3.14.0 -y
conda activate anmap
python -m pip install --upgrade pip

# habitat
conda install habitat-sim=0.2.5 withbullet -c conda-forge -c aihabitat -y

# install PyTorch 2.0.1 with CUDA 11.8:
pip uninstall torch torchvision functorch tinycudann
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

# install tiny-cuda-nn
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# requirements for rotorpy
pip install -e planning/rotorpy

# other requirements
conda install scikit-image PyYAML imageio tqdm scipy rich
pip install lpips opencv-python
```
### Download habitat data
```
# extract to data/scene_datasets/
https://drive.google.com/file/d/1qXl0iTlKawCXpJ1QJDM-IljmlUVXuyNp/view?usp=drive_link

# you can do so by gdown
pip install gdown==4.6.0
mkdir -p data/scene_datasets/
cd data/scene_datasets/
gdown https://drive.google.com/uc?id=1qXl0iTlKawCXpJ1QJDM-IljmlUVXuyNp
unzip hssd-hab.zip
```

### Run pipeline
```
# scene 1
python scripts/pipeline.py --sem-num 29 --habitat-scene 102344250

# scene 2
python scripts/pipeline.py --sem-num 29 --habitat-scene 102344529

# scene 3
python scripts/pipeline.py --sem-num 29 --habitat-scene 102344280
```
Data will be saved in the `data/habitat_collection/`:
```
/timestamp
	/checkpoints: the saved NeRF model checkpoints
	/maps: the 2D occupancy maps from NeRF
	/test: save test data
	/train: save train data collected during active perception
	/vis: images, predictions, fpv, and tpv for videos
	errors.npy stores the evaluation errors during active perception
	uncertainty.npy stores the predictive information during active perception
```

## Citation
```
@inproceedings{siming2024active,
  title={Active perception using neural radiance fields},
  author={Siming, H and Hsu, Christopher D and Ong, Dexter and Shao, Yifei Simon and Chaudhari, Pratik},
  booktitle={2024 American Control Conference (ACC)},
  pages={4353--4358},
  year={2024},
  organization={IEEE}
}
```
