#!/bin/bash
MODE=$1
DEVICE=$2

SCENES_DNERF="bouncingballs hellwarrior hook jumpingjacks lego mutant standup trex"
SCENES_HYPERNERF="3dprinter broom chicken peel-banana"

# https://github.com/pytorch/pytorch/issues/22866
# Need to set `OMP_NUM_THREADS` to prevent 2000% CPU usage

case $MODE in

  dnerf-vanilla)
    echo "D-NeRF dataset: vanilla"
    for scene in $SCENES_DNERF; do
      echo $scene
      OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$DEVICE python run.py --config configs/nerf-small/$scene.py
    done

    for scene in $SCENES_DNERF; do
      echo $scene
      OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$DEVICE python run.py --config configs/nerf-small/$scene.py --render_test --render_only --eval_psnr --eval_lpips_vgg
    done
    ;;

  dnerf-nerfacc-occgrid)
    echo "D-NeRF dataset: nerfacc (occgrid)"
    for scene in $SCENES_DNERF; do
      echo $scene
      OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$DEVICE python run.py --config configs/nerf-small-nerfacc-occgrid/$scene.py
    done

    for scene in $SCENES_DNERF; do
      echo $scene
      OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$DEVICE python run.py --config configs/nerf-small-nerfacc-occgrid/$scene.py --render_test --render_only --eval_psnr --eval_lpips_vgg
    done
    ;;

  hypernerf-vanilla)
    echo "Hyper-NeRF dataset: vanilla"
    for scene in $SCENES_HYPERNERF; do
      echo $scene
      OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$DEVICE python run.py --config configs/vrig_dataset/$scene.py
    done

    for scene in $SCENES_HYPERNERF; do
      echo $scene
      OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$DEVICE python run.py --config configs/vrig_dataset/$scene.py --render_test --render_only --eval_psnr --eval_lpips_vgg
    done
    ;;

  hypernerf-nerfacc-occgrid)
    echo "Hyper-NeRF dataset: nerfacc (occgrid)"
    for scene in $SCENES_HYPERNERF; do
      echo $scene
      OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$DEVICE python run.py --config configs/vrig_dataset-nerfacc-occgrid/$scene.py
    done

    for scene in $SCENES_HYPERNERF; do
      echo $scene
      OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$DEVICE python run.py --config configs/vrig_dataset-nerfacc-occgrid/$scene.py --render_test --render_only --eval_psnr --eval_lpips_vgg
    done
    ;;

  hypernerf-nerfacc-propnet)
    echo "Hyper-NeRF dataset: nerfacc (propnet)"
    for scene in $SCENES_HYPERNERF; do
      echo $scene
      OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$DEVICE python run.py --config configs/vrig_dataset-nerfacc-propnet/$scene.py
    done

    for scene in $SCENES_HYPERNERF; do
      echo $scene
      OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$DEVICE python run.py --config configs/vrig_dataset-nerfacc-propnet/$scene.py --render_test --render_only --eval_psnr --eval_lpips_vgg
    done
    ;;
esac
