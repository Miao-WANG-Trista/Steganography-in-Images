from flask import Flask,render_template,request
from datetime import datetime
import time
import os,glob
import sys
sys.path.insert(1,'./')
from predict.predict_folder_richmodels import richmodels
from predict.predict_folder_pytorch import pytorch_predict
from predict.group_experiments import group
from predict.Catboost_predict import catboost_predict

basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html',result=2)

@app.route('/predict', methods=['POST'])
def predict():
    time_start = time.time()
    test_single_image = eval(request.form['test_single_image'])
    folder = request.form['folder']
    test_single_image = True if folder=="" else False
    subset = request.form['subset']
    device = request.form['device']
    num_workers = int(request.form['num_workers'])
    batch_size = int(request.form['batch_size'])
    path0 = 'models_predictions/'+ subset
    to_delete_files = glob.glob(os.path.join(path0, '*.csv'))
    [(os.remove(f), to_delete_files.remove(f)) for f in to_delete_files]
    if test_single_image:
        image_size =1
        f = request.files.get('file')

        filename = datetime.now().strftime("%Y%m%d%H%M%S") + "." + "JPG"
        print('new filename is: ', filename)
        file_path = basedir +"/uploaded_images/"

        os.makedirs(file_path, exist_ok=True)
        f.save(file_path+filename)
        folder = file_path + filename
    else:
        DATA_ROOT_PATH = os.environ.get('DATA_ROOT_PATH')
        filename = DATA_ROOT_PATH+'/'+subset+'Test_qf_dicts.p'
        try:
            os.remove(filename)
        except OSError:
            pass
        image_size = len(os.listdir(os.path.join(DATA_ROOT_PATH,folder)))
        print('We are processing {} images'.format(image_size))
    if subset == '3Algorithms':
        print('yes,I am here, entering prediction process for 3Algorithms')
        richmodels(folder=folder,experiment='JRM',checkpoint='weights/rich_models/QF75_JRM_Y_ensemble_v7.mat',test_single_image=test_single_image)

        richmodels(folder=folder, experiment='DCTR', quality_factor=75,
                   checkpoint='weights/rich_models/QF75_DCTR_Y_ensemble_v7.mat', test_single_image=test_single_image)
        if not test_single_image:
            richmodels(folder=folder, experiment='JRM', quality_factor=90,
                       checkpoint='weights/rich_models/QF90_JRM_Y_ensemble_v7.mat', test_single_image=test_single_image)
            richmodels(folder=folder, experiment='JRM', quality_factor=95,
                       checkpoint='weights/rich_models/QF95_JRM_Y_ensemble_v7.mat', test_single_image=test_single_image)
            richmodels(folder=folder, experiment='DCTR', quality_factor=90,
                       checkpoint='weights/rich_models/QF90_DCTR_Y_ensemble_v7.mat',
                       test_single_image=test_single_image)
            richmodels(folder=folder, experiment='DCTR', quality_factor=95,
                       checkpoint='weights/rich_models/QF95_DCTR_Y_ensemble_v7.mat',
                       test_single_image=test_single_image)
        pytorch_predict(folder=folder,model='efficientnet-b4',experiment='efficientnet_b4_NR',decoder='NR',checkpoint='weights/efficientnet_b4_NR_mish/best-checkpoint-017epoch.bin', test_single_image=test_single_image, device=device,num_workers=num_workers,batch_size=batch_size)
        pytorch_predict(folder=folder,model='efficientnet-b5',experiment='efficientnet_b5_NR',decoder='NR',checkpoint='weights/efficientnet_b5_NR_mish/best-checkpoint-018epoch.bin', test_single_image=test_single_image, device=device,num_workers=num_workers,batch_size=batch_size)
        pytorch_predict(folder=folder,model='mixnet_xl',experiment='mixnet_xL_NR_mish',decoder='NR',checkpoint='weights/mixnet_xL_NR_mish/best-checkpoint-021epoch.bin',test_single_image=test_single_image, surgery=1, device=device,num_workers=num_workers,batch_size=batch_size)
        pytorch_predict(folder=folder,model='efficientnet-b2',experiment='efficientnet_b2_NR',decoder='NR',checkpoint='weights/efficientnet_b2/NR/best-checkpoint-028epoch.bin', test_single_image=test_single_image,device=device,num_workers=num_workers,batch_size=batch_size)
        pytorch_predict(folder=folder,model='efficientnet-b2',experiment='efficientnet_b2_R',checkpoint='weights/efficientnet_b2/R/best-checkpoint-028epoch.bin',test_single_image=test_single_image,  device=device,num_workers=num_workers,batch_size=batch_size)
        pytorch_predict(folder=folder,model='mixnet_s',experiment='mixnet_S_R_seed0',checkpoint='weights/mixnet_S/R_seed0/best-checkpoint-033epoch.bin',test_time_augmentation=1, test_single_image=test_single_image, device=device,num_workers=num_workers,batch_size=batch_size)
        pytorch_predict(folder=folder,model='mixnet_s',experiment='mixnet_S_R_seed1',checkpoint='weights/mixnet_S/R_seed1/best-checkpoint-035epoch.bin',test_time_augmentation=1, test_single_image=test_single_image, device=device,num_workers=num_workers,batch_size=batch_size)
        pytorch_predict(folder=folder,model='mixnet_s',experiment='mixnet_S_R_seed2',checkpoint='weights/mixnet_S/R_seed2/best-checkpoint-036epoch.bin',test_time_augmentation=1, test_single_image=test_single_image, device=device,num_workers=num_workers,batch_size=batch_size )
        pytorch_predict(folder=folder,model='mixnet_s',experiment='mixnet_S_R_seed3',checkpoint='weights/mixnet_S/R_seed3/best-checkpoint-038epoch.bin',test_time_augmentation=1, test_single_image=test_single_image, device=device,num_workers=num_workers,batch_size=batch_size)
        pytorch_predict(folder=folder,model='mixnet_s',experiment='mixnet_S_R_seed4',checkpoint='weights/mixnet_S/R_seed4/best-checkpoint-035epoch.bin',test_time_augmentation=1, test_single_image=test_single_image, device=device,num_workers=num_workers,batch_size=batch_size)
        pytorch_predict(folder=folder,model='mixnet_s',experiment='mixnet_S_NR',decoder='NR', checkpoint='weights/mixnet_S/NR/best-checkpoint-058epoch.bin',test_time_augmentation=1, test_single_image=test_single_image, device=device,num_workers=num_workers,batch_size=batch_size)
        id = '0000'
        group(id=id, test_single_image=test_single_image)
        parent_dir = basedir + '/models_predictions/' + subset + '/'
        result = catboost_predict(zoo_file=parent_dir + 'probabilities_zoo_Test_' + id + '.csv',
                                  test_single_image=test_single_image)

    else:
        print('yes,I am here, entering prediction process for nsf5')

        richmodels(folder=folder, experiment='JRM',
                   checkpoint='weights/nsf5/rich_models_for_nsf5/JRM_Y_ensemble_v7.mat', subset='for_nsf5', test_single_image=test_single_image)
        richmodels(folder=folder, experiment='DCTR',
                   checkpoint='weights/nsf5/rich_models_for_nsf5/DCTR_Y_ensemble_v7.mat', subset='for_nsf5',
                   test_single_image=test_single_image)

        pytorch_predict(folder=folder, model='efficientnet-b4', experiment='efficientnet_b4',
                       checkpoint='weights/nsf5/efficientnet_b4/best-checkpoint-001epoch.bin', subset='for_nsf5',
                        test_single_image=test_single_image, device=device, num_workers=num_workers,
                        batch_size=batch_size)
        pytorch_predict(folder=folder, model='efficientnet-b5', experiment='efficientnet_b5',
                        checkpoint='weights/nsf5/efficientnet_b5/best-checkpoint-003epoch.bin',subset='for_nsf5',
                        test_single_image=test_single_image, device=device, num_workers=num_workers,
                        batch_size=batch_size)
        pytorch_predict(folder=folder, model='mixnet_xl', experiment='mixnet_xl',
                        checkpoint='weights/nsf5/mixnet_xL_R/last-checkpoint.bin',
                        test_single_image=test_single_image, surgery=1, device=device, num_workers=num_workers,subset='for_nsf5',
                        batch_size=batch_size)
        pytorch_predict(folder=folder, model='efficientnet-b2', experiment='efficientnet_b2',
                        checkpoint='weights/nsf5/efficientnet_b2/best-checkpoint-003epoch.bin',subset='for_nsf5',
                        test_single_image=test_single_image, device=device, num_workers=num_workers,
                        batch_size=batch_size)
        pytorch_predict(folder=folder,model='mixnet_s',experiment='mixnet_S',checkpoint='weights/nsf5/mixnet_s_R/best-checkpoint-000epoch.bin',
                        test_time_augmentation=1, test_single_image=test_single_image, device=device,num_workers=num_workers, batch_size=batch_size,subset='for_nsf5')

        id = '0000'
        group(id=id, test_single_image=test_single_image, subset=subset)
        parent_dir = basedir + '/models_predictions/' + subset + '/'
        result = catboost_predict(zoo_file=parent_dir + 'probabilities_zoo_Test_' + id + '.csv',
                                  test_single_image=test_single_image,subset=subset,weights_path='weights/nsf5/catboost/best_catboost.cmb')

    time_end = time.time()
    running_time = time_end - time_start
    return render_template('index.html', message = 'We processed %d images, time cost : %.3f sec' %(image_size,running_time), prediction_text='Detected result is {}'.format(result) if test_single_image else 'Detection result is saved.')

def main():
    app.run(host='0.0.0.0',port=8000,debug=True)

if __name__ == "__main__":
    main()


