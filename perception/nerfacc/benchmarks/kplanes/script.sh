#!/bin/bash
MODE=$1
DEVICE=$2

SCENES_DNERF="bouncingballs hellwarrior hook jumpingjacks lego mutant standup trex"

case $MODE in

  dnerf-vanilla)
    echo "D-NeRF dataset: vanilla"
    for scene in $SCENES_DNERF; do
      echo $scene
      CUDA_VISIBLE_DEVICES=$DEVICE python -m plenoxels.main \
        --config-path plenoxels/configs/final/D-NeRF/dnerf_hybrid.py \
        --trainval \
        --data-dir data/dnerf/data/$scene \
        expname="${scene}_hybrid"
    done
    ;;

  dnerf-nerfacc-occgrid)
    echo "D-NeRF dataset: nerfacc (occgrid)"
    for scene in $SCENES_DNERF; do
      echo $scene
      CUDA_VISIBLE_DEVICES=$DEVICE python -m plenoxels.main \
        --config-path plenoxels/configs/final/D-NeRF/dnerf_hybrid.py \
        --trainval \
        --data-dir data/dnerf/data/$scene \
        expname="${scene}_hybrid_nerfacc" occ_grid_reso=128 occ_step_size=0.004
    done
    ;;
esac