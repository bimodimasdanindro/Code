Code File Information

**Dataload_segmentation.ipynb** 
for data loading and integrating it with the annotated data

**S1_cae_unet_convlstm_repaired**
for running CAE U-Net ConvLSTM and CAE U-Net BiConvLSTM (with --bidirectional input specified)

**gao_semi_cae_convlstm.py**
for running the Gao et al. model implementation without a skip connection

**S2_cae_unet_convlstm_global.py**
for running Stage~2 with code implementation with global sample mean and standard deviation normalisation

**S2_cae_unet_convlstm_meanstd_fold.py**
for running Stage~2 with code implementation with normalisation on the training set of each fold's mean and standard deviation

**S2_cae_unet_convlstm_v3.py**
for running Stage~2 with code implementation with normalisation on the fold's 1 mean and standard deviation
