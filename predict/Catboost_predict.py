import pandas as pd
from sklearn.preprocessing import StandardScaler
import argparse
import os
from catboost import CatBoostClassifier, Pool
import numpy as np
import pickle

def main():

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--zoo-file', type=str, default='models_predictions/probabilities_zoo_lb.csv', help='path to zoo file')
    arg('--weights_path', type=str, default='weights/catboost/best_catboost.cmb', help='path to catboost weights')
    arg('--n-splits', type=int, default=10 , help='num CV splits')
    arg("--test_single_image", help='test single image', action='store_true')

    args = parser.parse_args()
    df = pd.read_csv(args.zoo_file,index_col=0)
    model = CatBoostClassifier()
    model = model.load_model(args.weights_path)

    df = df[['NAME',
             'efficientnet_b5_NR_pc', 'efficientnet_b5_NR_pjm',
             'efficientnet_b5_NR_pjuni', 'efficientnet_b5_NR_puerd', 'DCTR',
             'mixnet_S_pc', 'mixnet_S_pjm', 'mixnet_S_pjuni', 'mixnet_S_puerd',
             'mixnet_xL_NR_mish_pc', 'mixnet_xL_NR_mish_pjm',
             'mixnet_xL_NR_mish_pjuni', 'mixnet_xL_NR_mish_puerd', 'JRM', 'QF',
             'efficientnet_b4_NR_pc', 'efficientnet_b4_NR_pjm',
             'efficientnet_b4_NR_pjuni', 'efficientnet_b4_NR_puerd',
             'efficientnet_b2_pc', 'efficientnet_b2_pjm',
             'efficientnet_b2_pjuni', 'efficientnet_b2_puerd',
            ]]

    df['QF'] = df['QF'].astype('int').astype('str') # 'QF' should be converted into str cuz it's categorical feature rather than numerical
    scaler = StandardScaler() # apply standardization on numerical columns to avoid the influence of scale
    num_cols = df.columns[df.dtypes.apply(lambda c: np.issubdtype(c, np.number))]
    df[num_cols] = scaler.fit_transform(df[num_cols])
    df_features = df.drop(columns=['NAME'])
    scores = model.predict(df_features)
    sub = pd.DataFrame({"Id": df.NAME, "Label": scores})
    if args.test_single_image:
        print('this image is', 'malicous' if scores[0]==1 else 'innocent')
    else:
        with open('models_predictions/out_of_bounds_Test.p', 'rb') as handle:
            oor = pickle.load(handle)

        for im_name in oor:
            sub.loc[sub.Id == im_name, 'Label'] = 1.01

        sub.to_csv('final_results.csv')
def catboost_predict(zoo_file='models_predictions/probabilities_zoo_lb.csv',weights_path='weights/catboost/best_catboost.cmb',n_splits=10,test_single_image=False,subset='3Algorithms'):

    df = pd.read_csv(zoo_file,index_col=0)
    model = CatBoostClassifier()
    model = model.load_model(weights_path)
    if subset == '3Algorithms':
        df = df[['NAME',
                 'efficientnet_b5_NR_pc', 'efficientnet_b5_NR_pjm',
                 'efficientnet_b5_NR_pjuni', 'efficientnet_b5_NR_puerd', 'DCTR',
                 'mixnet_S_pc', 'mixnet_S_pjm', 'mixnet_S_pjuni', 'mixnet_S_puerd',
                 'mixnet_xL_NR_mish_pc', 'mixnet_xL_NR_mish_pjm',
                 'mixnet_xL_NR_mish_pjuni', 'mixnet_xL_NR_mish_puerd', 'JRM', 'QF',
                 'efficientnet_b4_NR_pc', 'efficientnet_b4_NR_pjm',
                 'efficientnet_b4_NR_pjuni', 'efficientnet_b4_NR_puerd',
                 'efficientnet_b2_pc', 'efficientnet_b2_pjm',
                 'efficientnet_b2_pjuni', 'efficientnet_b2_puerd',
                ]]
    else:
        df = df[['NAME',
                 'mixnet_xl_pc','mixnet_xl_pnsf5','efficientnet_b5_pc','efficientnet_b5_pnsf5','DCTR',
                 'efficientnet_b4_pc', 'efficientnet_b4_pnsf5','efficientnet_b2_pc','efficientnet_b2_pnsf5',
                 'mixnet_S_pc','mixnet_S_pnsf5','JRM','QF',
                 ]]

    df['QF'] = df['QF'].astype('int').astype('str') # 'QF' should be converted into str cuz it's categorical feature rather than numerical
    scaler = StandardScaler() # apply standardization on numerical columns to avoid the influence of scale
    num_cols = df.columns[df.dtypes.apply(lambda c: np.issubdtype(c, np.number))]
    df[num_cols] = scaler.fit_transform(df[num_cols])
    df_features = df.drop(columns=['NAME'])
    scores = model.predict(df_features)
    sub = pd.DataFrame({"Id": df.NAME, "Label": scores})
    if test_single_image:
        result='malicous' if scores[0]==1 else 'innocent'
        return result
    else:
        with open('models_predictions/out_of_bounds_Test.p', 'rb') as handle:
            oor = pickle.load(handle)

        for im_name in oor:
            sub.loc[sub.Id == im_name, 'Label'] = 1.01

        sub.to_csv('final_results.csv')
        print('prediction results is done in final_results.csv')
        return None
if __name__ == "__main__":
    main()
