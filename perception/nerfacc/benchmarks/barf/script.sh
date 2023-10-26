#!/bin/bash
MODE=$1
DEVICE=$2

SCENES_NERFSYN="chair drums ficus hotdog lego materials mic ship"

case $MODE in

  nerfsyn-vanilla)
    echo "NeRF-Synthetic dataset: vanilla"
    for scene in $SCENES_NERFSYN; do
      echo $scene
      CUDA_VISIBLE_DEVICES=$DEVICE python train.py --group=vanilla --model=barf --yaml=barf_blender --name=$scene --data.scene=$scene --barf_c2f=[0.1,0.5] --visdom!
    done

    for scene in $SCENES_NERFSYN; do
      echo $scene
      CUDA_VISIBLE_DEVICES=$DEVICE python train.py --group=vanilla --model=barf --yaml=barf_blender --name=$scene --data.scene=$scene --data.val_sub= --resume --visdom!
    done
    ;;

  nerfsyn-nerfacc-occgrid)
    echo "NeRF-Synthetic dataset: nerfacc (occgrid)"
    for scene in $SCENES_NERFSYN; do
      echo $scene
      CUDA_VISIBLE_DEVICES=$DEVICE python train.py --group=nerfacc --model=barf --yaml=barf_blender_nerfacc --name=$scene --data.scene=$scene --barf_c2f=[0.1,0.5] --visdom!
    done

    for scene in $SCENES_NERFSYN; do
      echo $scene
      CUDA_VISIBLE_DEVICES=$DEVICE python evaluate.py --group=nerfacc --model=barf --yaml=barf_blender_nerfacc --name=$scene --data.scene=$scene --data.val_sub= --resume --visdom!
    done
    ;;
esac