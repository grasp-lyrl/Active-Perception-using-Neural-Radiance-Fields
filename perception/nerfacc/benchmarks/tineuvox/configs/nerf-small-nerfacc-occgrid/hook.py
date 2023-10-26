_base_ = './default.py'

expname = 'small/dnerf_hook-400-nerfacc-occgrid'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='./data_dnerf/hook',
    dataset_type='dnerf',
    white_bkgd=True,
    aabb=(-1.3, -1.3, -1.3, 1.3, 1.3, 1.3),
)

model_and_render = dict(
    occ_grid_reso=128,
    occ_ema_decay=0.99,
)