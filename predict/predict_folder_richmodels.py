import os
import numpy as np 
import argparse
from tqdm import tqdm
import pandas as pd
import sys
import pickle
import jpegio as jio
from oct2py import octave
sys.path.insert(1,'./')
from train_module.tools.jpeg_utils import *
from pathlib import Path
octave.addpath('rich_models/')

def main():
    DATA_ROOT_PATH = os.environ.get('DATA_ROOT_PATH')
    parser = argparse.ArgumentParser("Predict Test images using rich models")
    arg = parser.add_argument
    arg('--folder', type=str, default='Test/', help='path to test images')
    arg('--experiment', type=str, default='DCTR', help='specific model experiment name')
    arg('--checkpoint', type=str, default='' , help='path to checkpoint')
    arg('--quality-factor', type=int, default=75 , help='quality factor')
    arg('--output', type=str, default='models_predictions/', help='output folder')
    arg('--subset', type=str, default='LB' , help='A subset of the folder? train_module, test or val')
    arg("--test_single_image", help='test single image', action='store_true')
    
    args = parser.parse_args()
    os.makedirs(os.path.join(args.output, args.subset), exist_ok=True)

    if not args.test_single_image:
        folder = os.path.join(DATA_ROOT_PATH, args.folder)
        if args.subset == 'LB':
            names = os.listdir(folder)
            test_qf_dicts_path = os.path.join(DATA_ROOT_PATH, 'Test_qf_dicts.p')
            if not os.path.exists(test_qf_dicts_path):
                (names_qf, qf_names) = get_qf_dicts(folder, names)
                with open(test_qf_dicts_path, 'wb') as handle:
                    pickle.dump((names_qf, qf_names), handle)
            else:
                with open(test_qf_dicts_path, 'rb') as handle:
                    (names_qf, qf_names) = pickle.load(handle)

            IL = qf_names[args.quality_factor]

        else:
            with open('./IL_'+args.subset+'_'+str(args.quality_factor)+'.p', 'rb') as handle:
                IL = pickle.load(handle)

        if args.experiment == 'DCTR':
            f = octave.DCTR
        elif args.experiment == 'JRM':
            f = octave.JRM
        filelist = os.listdir(folder)

        votes = []
        for im_name in tqdm(IL, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
            tmp = jio.read(os.path.join(folder, im_name))
            feature = f(tmp.coef_arrays[0], tmp.quant_tables[0])
            # coef_arrays is a list of numpy.ndarray objects that represent DCT coefficients of YCbCr channels in JPEG
            # quant_tables is a list of numpy.ndarray objects that represent the quantization tables in JPEG.
            fld_ensemble_prediction = octave.ensemble_testing(feature, args.checkpoint)
            votes.append(fld_ensemble_prediction['votes'])

    else:
        if args.experiment == 'DCTR':
            f = octave.DCTR
        elif args.experiment == 'JRM':
            f = octave.JRM


        votes = []

        tmp = jio.read(os.path.join(args.folder))
        feature = f(tmp.coef_arrays[0], tmp.quant_tables[0])
        # coef_arrays is a list of numpy.ndarray objects that represent DCT coefficients of YCbCr channels in JPEG
        # quant_tables is a list of numpy.ndarray objects that represent the quantization tables in JPEG.
        fld_ensemble_prediction = octave.ensemble_testing(feature, args.checkpoint)
        votes.append(fld_ensemble_prediction['votes'])
        
    pred_dataframe = pd.DataFrame(columns=['NAME', args.experiment])

    file_name = args.folder.split("/")[-1]
    pred_dataframe['NAME'] = IL if not args.test_single_image else [file_name]
    pred_dataframe[args.experiment] = votes
    output_path = Path(os.path.join(args.output, args.subset, 'QF'+str(args.quality_factor)+'_'+args.experiment+'_votes_Test.csv'))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_dataframe.to_csv(output_path)
    
if __name__ == "__main__":
    main()