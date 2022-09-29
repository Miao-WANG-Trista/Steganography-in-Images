import argparse
import sys
import numpy as np
import os
import pickle
import jpegio as jio
from oct2py import octave
from tqdm import tqdm
octave.addpath('rich_models/')
#sys.path.insert(1,'./')


def main():
    
    # DATA_ROOT_PATH = os.environ.get('DATA_ROOT_PATH')
    parser = argparse.ArgumentParser("Trains FLD ensemble from feature maps")
    arg = parser.add_argument
    arg('--model', type=str, default='DCTR', help='model name')
    arg('--features-folder', type=str, default='train_module/features/', help='model name')
    # arg('--quality-factor', type=int, default=75, help='quality factor')
    arg('--output', type=str, default='weights/rich_models/for_nsf5', help='model name')
    
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
        
    features_cover = np.load(os.path.join(args.features_folder,args.model+'_features_Cover.npy'))
    features_stego = np.zeros_like(features_cover)
    num_images = np.shape(features_stego)[0]
    f = np.load(os.path.join(args.features_folder,args.model+'_features_nsf5.npy'))
    features_stego[:num_images,:] = f[:num_images,:]

    del f
    save_path = os.path.join(args.output, args.model+'_Y_ensemble_v7.mat')
    _ = octave.ensemble_training(features_cover, features_stego, save_path)

if __name__ == "__main__":
    main()