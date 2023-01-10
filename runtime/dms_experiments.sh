
# python single_pdb_inference.py \
#                                --experiment_dir runs/so3_convnet_lmax=6__v1 \
#                                --pdb_filepath /gscratch/spe/gvisan01/dms_data/BLAT_ECOLX_Palzkill2012/1xpb.pdb \
#                                --output_filepath runs/so3_convnet_lmax=6__v1/1xpb-lowest_valid_loss_model.npz \
#                                --model_name lowest_valid_loss_model

python dms_comparison.py \
                         --predictions_filepath runs/so3_convnet_lmax=6__v1/1xpb-lowest_valid_loss_model.npz \
                         --dms_input_filepath /gscratch/spe/gvisan01/dms_data/BLAT_ECOLX_Palzkill2012/output_BLAT_ECOLX_Palzkill2012.csv \
                         --dms_output_filepath runs/so3_convnet_lmax=6__v1/1xpb-lowest_valid_loss_model.csv \
                         --dms_column ddG_stat \
                         --dms_filepath runs/so3_convnet_lmax=6__v1/1xpb-ddG_stat-lowest_valid_loss_model.png

