_base_ = './hyper_default.py'

expname = 'vrig/base-3dprinter-nerfacc-occgrid'
basedir = './logs/vrig_data'

data = dict(
    datadir='./vrig-3dprinter',
    dataset_type='hyper_dataset',
    white_bkgd=False,
)

model_and_render = dict(
    occ_grid_reso=128,
    occ_ema_decay=0.99,
    occ_thre=0.05,
)