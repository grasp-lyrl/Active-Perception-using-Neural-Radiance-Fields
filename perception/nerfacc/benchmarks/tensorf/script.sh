#!/bin/bash
MODE=$1
DEVICE=$2

SCENES_NERFSYN="chair drums ficus hotdog lego materials mic ship"
SCENES_TT="Barn Caterpillar Family Truck"

# https://github.com/pytorch/pytorch/issues/22866
# Need to set `OMP_NUM_THREADS` to prevent 2000% CPU usage

case $MODE in

  nerfsyn-vanilla)
    echo "NeRF-Synthetic dataset: vanilla"
    for scene in $SCENES_NERFSYN; do
      echo $scene
      OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$DEVICE python train.py --config configs/lego.txt --datadir ./data/nerf_synthetic/$scene --expname $scene
    done
    ;;

  nerfsyn-nerfacc-occgrid)
    echo "NeRF-Synthetic: nerfacc (occgrid)"
    for scene in $SCENES_NERFSYN; do
      echo $scene
      OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$DEVICE python train.py --config configs/lego.txt --datadir ./data/nerf_synthetic/$scene --expname $scene --occ_grid_reso 128 --basedir ./log_nerfacc
    done
    ;;

  tt-vanilla)
    echo "TT dataset: vanilla"
    for scene in $SCENES_TT; do
      echo $scene
      OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$DEVICE python train.py --config configs/truck.txt --datadir ./data/TanksAndTemple/$scene --expname $scene
    done
    ;;

  tt-nerfacc-occgrid)
    echo "TT dataset: nerfacc (occgrid)"
    for scene in $SCENES_TT; do
      echo $scene
      OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$DEVICE python train.py --config configs/truck.txt --datadir ./data/TanksAndTemple/$scene  --expname $scene --occ_grid_reso 128 --basedir ./log_nerfacc
    done
    ;;
esac
