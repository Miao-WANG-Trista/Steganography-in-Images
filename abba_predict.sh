export DATA_ROOT_PATH=./Test_images
#python3 download_weights.py
python3 predict/predict_folder_outofbounds.py
python3 predict/predict_folder_richmodels.py --experiment JRM --quality-factor 75 --checkpoint weights/rich_models/QF75_JRM_Y_ensemble_v7.mat &
python3 predict/predict_folder_richmodels.py --experiment JRM --quality-factor 90 --checkpoint weights/rich_models/QF90_JRM_Y_ensemble_v7.mat &
python3 predict/predict_folder_richmodels.py --experiment JRM --quality-factor 95 --checkpoint weights/rich_models/QF95_JRM_Y_ensemble_v7.mat &
python3 predict/predict_folder_richmodels.py --experiment DCTR --quality-factor 75 --checkpoint weights/rich_models/QF75_DCTR_Y_ensemble_v7.mat &
python3 predict/predict_folder_richmodels.py --experiment DCTR --quality-factor 90 --checkpoint weights/rich_models/QF90_DCTR_Y_ensemble_v7.mat &
python3 predict/predict_folder_richmodels.py --experiment DCTR --quality-factor 95 --checkpoint weights/rich_models/QF95_DCTR_Y_ensemble_v7.mat


python3 predict/predict_folder_pytorch.py --model efficientnet-b4 --experiment efficientnet_b4_NR --decoder NR --checkpoint weights/efficientnet_b4_NR_mish/best-checkpoint-017epoch.bin
python3 predict/predict_folder_pytorch.py --model efficientnet-b5 --experiment efficientnet_b5_NR --decoder NR --checkpoint weights/efficientnet_b5_NR_mish/best-checkpoint-018epoch.bin
python3 predict/predict_folder_pytorch.py --model mixnet_xl --experiment mixnet_xL_NR_mish --surgery 1 --decoder NR --checkpoint weights/mixnet_xL_NR_mish/best-checkpoint-021epoch.bin
python3 predict/predict_folder_pytorch.py --model efficientnet-b2 --experiment efficientnet_b2_NR --decoder NR--checkpoint weights/efficientnet_b2/NR/best-checkpoint-028epoch.bin
python3 predict/predict_folder_pytorch.py --model efficientnet-b2 --experiment efficientnet_b2_R --checkpoint weights/efficientnet_b2/R/best-checkpoint-028epoch.bin
python3 predict/predict_folder_pytorch.py --model mixnet_s --test-time-augmentation 1 --experiment mixnet_S_R_seed0 --checkpoint weights/mixnet_S/R_seed0/best-checkpoint-033epoch.bin
python3 predict/predict_folder_pytorch.py --model mixnet_s --test-time-augmentation 1 --experiment mixnet_S_R_seed1 --checkpoint weights/mixnet_S/R_seed1/best-checkpoint-035epoch.bin
python3 predict/predict_folder_pytorch.py --model mixnet_s --test-time-augmentation 1 --experiment mixnet_S_R_seed2 --checkpoint weights/mixnet_S/R_seed2/best-checkpoint-036epoch.bin
python3 predict/predict_folder_pytorch.py --model mixnet_s --test-time-augmentation 1 --experiment mixnet_S_R_seed3 --checkpoint weights/mixnet_S/R_seed3/best-checkpoint-038epoch.bin
python3 predict/predict_folder_pytorch.py --model mixnet_s --test-time-augmentation 1 --experiment mixnet_S_R_seed4 --checkpoint weights/mixnet_S/R_seed4/best-checkpoint-035epoch.bin
python3 predict/predict_folder_pytorch.py --model mixnet_s --test-time-augmentation 1 --experiment mixnet_S_NR --decoder NR --checkpoint weights/mixnet_S/NR/best-checkpoint-058epoch.bin

python3 predict/group_experiments.py
python3 predict/Catboost_predict.py --zoo-file models_predictions/LB/probabilities_zoo_Test_0000.csv