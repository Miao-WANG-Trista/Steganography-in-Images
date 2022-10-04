import pandas as pd
from datetime import date
from functools import reduce
import argparse
import pickle
import os
import glob


def main():
    
    DATA_ROOT_PATH = os.environ.get('DATA_ROOT_PATH')
    parser = argparse.ArgumentParser("Group all zoo predictions")
    arg = parser.add_argument
    arg('--output', type=str, default='models_predictions/', help='output folder')

    arg('--id', type=str, default='0000', help='output folder')
    arg('--subset', type=str, default='3Algorithms', help='the folder for three algorithms or nsf5?')
    arg("--test_single_image", help='test single image', action='store_true')
    
    args = parser.parse_args()
    #today = date.today()
    #d = today.strftime('%m%d')
    d = args.id
    folder_subpath = 'Test'
    
    all_files = glob.glob(os.path.join(args.output,'*'+folder_subpath+'.csv'))

    def group_seeds(grouped_experiment, str_filter):
        files = [f for f in all_files if str_filter in f]
        dfs = [pd.read_csv(f, index_col=0) for f in files] if len(files) !=1 else pd.read_csv(files,index_col=0)
        probabilities_zoo_lb = reduce(lambda left,right: pd.merge(left,right), dfs) if len(files)!=1 else dfs
        columns = list(probabilities_zoo_lb.columns)
        columns.remove('NAME')
        experiments = list(set([c.split('_p')[0] for c in columns[:]]))
        if args.subset == '3Algorithms':
            columns_name = [grouped_experiment + '_pc', grouped_experiment + '_pjm', grouped_experiment+ '_pjuni',
                       grouped_experiment + '_puerd']
            column_list = ['NAME', grouped_experiment + '_pc', grouped_experiment + '_pjm', grouped_experiment + '_pjuni',
                           grouped_experiment + '_puerd']

        elif args.subset == 'for_nsf5':
            columns = [args.experiment + '_pc', args.experiment + '_pnsf5']
            column_list = ['NAME', args.experiment + '_pc', args.experiment + '_pnsf5']
        probabilities_zoo_lb2 = pd.DataFrame(columns=column_list)
        probabilities_zoo_lb2.NAME = probabilities_zoo_lb.NAME
        grouped_probas = 0
        for exp in experiments:
            grouped_probas += probabilities_zoo_lb[columns_name].values

        grouped_probas /= len(experiments)
        probabilities_zoo_lb2[columns_name] = grouped_probas
        [(os.remove(f), all_files.remove(f)) for f in files]
        return probabilities_zoo_lb2, grouped_experiment+'_probabilities'

    probabilities_zoo_lb_b2, grouped_experiment = group_seeds('efficientnet_b2', 'b2')
    probabilities_zoo_lb_b2.to_csv(os.path.join(args.output, grouped_experiment+'_'+folder_subpath+'.csv'))
    probabilities_zoo_lb_mixnet_S, grouped_experiment = group_seeds('mixnet_S', 'mixnet_S')
    probabilities_zoo_lb_mixnet_S.to_csv(os.path.join(args.output, grouped_experiment+'_'+folder_subpath+'.csv'))
    print('yes1')
    if args.subset == '3Algorithms':
        for exp in ['JRM_votes','DCTR_votes']:
            files = [os.path.join(args.output,'QF'+str(qf)+'_'+exp+'_'+folder_subpath+'.csv') for qf in [75,90,95]]
            dfs = [pd.read_csv(f, index_col=0) for f in files]
            df = pd.concat(dfs).reset_index(drop=True)
            df.to_csv(os.path.join(args.output, exp+'_'+folder_subpath+'.csv'))
            [(os.remove(f), all_files.remove(f)) for f in files]
    elif args.subset == 'for_nsf5':
        for exp in ['JRM_votes','DCTR_votes']:
            files = [os.path.join(args.output,'QF75'+'_'+exp+'_'+folder_subpath+'.csv')]
            dfs = [pd.read_csv(f, index_col=0) for f in files]
            df = pd.concat(dfs).reset_index(drop=True)
            df.to_csv(os.path.join(args.output, exp+'_'+folder_subpath+'.csv'))
            [(os.remove(f), all_files.remove(f)) for f in files]

        
    all_files = glob.glob(os.path.join(args.output,'*'+folder_subpath+'.csv'))        
    dfs = [pd.read_csv(f, index_col=0) for f in all_files]
    print('dfs',dfs)
    probabilities_zoo_lb = reduce(lambda left,right: pd.merge(left,right), dfs)
    [(os.remove(f), all_files.remove(f)) for f in all_files]

    if args.test_single_image:
        probabilities_zoo_lb['QF']=90 # needs to be changed according to the quality factor of the single image
    else:

        test_qf_dicts_path = os.path.join(DATA_ROOT_PATH, 'Test_qf_dicts.p')
        with open(test_qf_dicts_path, 'rb') as handle:
            (names_qf, qf_names) = pickle.load(handle)
        qf_df = pd.DataFrame.from_records(list(names_qf.items()),columns=['NAME','QF'])
        probabilities_zoo_lb = probabilities_zoo_lb.merge(qf_df)

    probabilities_zoo_lb.to_csv(os.path.join(args.output, 'probabilities_zoo_'+folder_subpath+'_'+d+'.csv'))


if __name__ == "__main__":
    main()