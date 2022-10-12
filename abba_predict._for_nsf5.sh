export DATA_ROOT_PATH=./Test_images
python3 predict/predict_folder_outofbounds.py --folder 'nsf5_images/for_predict/' --output 'models_predictions/for_nsf5/'
python3 predict/predict_folder_richmodels.py --folder 'nsf5_images/for_predict/' --experiment JRM --checkpoint weights/nsf5/rich_models_for_nsf5/JRM_Y_ensemble_v7.mat  --subset 'for_nsf5' --output 'models_predictions/for_nsf5/'&
python3 predict/predict_folder_richmodels.py --folder 'nsf5_images/for_predict/' --experiment DCTR --checkpoint weights/nsf5/rich_models_for_nsf5/DCTR_Y_ensemble_v7.mat --subset 'for_nsf5' --output 'models_predictions/for_nsf5/'



python3 predict/predict_folder_pytorch.py --model efficientnet-b4 --experiment efficientnet_b4 --checkpoint weights/nsf5/efficientnet_b4/best-checkpoint-001epoch.bin --subset 'for_nsf5' --folder '/nsf5_images/for_predict'
python3 predict/predict_folder_pytorch.py --model efficientnet-b5 --experiment efficientnet_b5 --checkpoint weights/nsf5/efficientnet_b5/best-checkpoint-003epoch.bin --subset 'for_nsf5' --folder '/nsf5_images/for_predict'
python3 predict/predict_folder_pytorch.py --model mixnet_xl --experiment mixnet_xl --surgery 1 --checkpoint weights/nsf5/mixnet_xL_R/last-checkpoint.bin --subset 'for_nsf5' --folder '/nsf5_images/for_predict'
python3 predict/predict_folder_pytorch.py --model efficientnet-b2 --experiment efficientnet_b2 --checkpoint weights/nsf5/efficientnet_b2/best-checkpoint-003epoch.bin --subset 'for_nsf5' --folder '/nsf5_images/for_predict'
python3 predict/predict_folder_pytorch.py --model mixnet_s --test-time-augmentation 1 --experiment mixnet_S --checkpoint weights/nsf5/mixnet_s_R/best-checkpoint-000epoch.bin --subset 'for_nsf5' --folder '/nsf5_images/for_predict'

python3 predict/group_experiments.py --output 'models_predictions/for_nsf5/' --subset 'for_nsf5'
python3 predict/Catboost_predict.py --zoo-file models_predictions/LB/probabilities_zoo_Test_$day.csv --catboost-file weights/catboost/best_catboost_TST_0.94001.cmb --version v26