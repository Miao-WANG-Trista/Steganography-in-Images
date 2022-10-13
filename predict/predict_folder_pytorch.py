import os
import numpy as np 
from torch import nn
import torch
import argparse
from tqdm import tqdm
import sys
sys.path.insert(1,'./')
# from apex.apex import amp
from torch.utils.data.sampler import SequentialSampler
sys.path.insert(1,'./')
from train_module.zoo.models import *
from train_module.zoo.surgery import *
from train_module.datafeeding.retriever import *
from train_module.tools.torch_utils import *
from pathlib import Path

def main():
    
    DATA_ROOT_PATH = os.environ.get('DATA_ROOT_PATH')
    parser = argparse.ArgumentParser("Predict Test images with TTA")
    arg = parser.add_argument
    arg('--folder', type=str, default='/Test', help='path to test images')
    arg('--model', type=str, default='efficientnet-b4', help='model name')
    arg('--experiment', type=str, default='efficientnet_b4_NR_mish', help='specific model experiment name')
    arg('--surgery', type=int, default=0, help='modification level')
    arg('--test-time-augmentation', type=int, default=4, help='TTA level')
    arg('--checkpoint', type=str, default='' , help='path to checkpoint')
    arg('--output', type=str, default='models_predictions/', help='output folder')
    arg('--decoder', type=str, default='R' , help='how to decode jpeg files, NR or R')
    arg('--fp16', type=int, default=0 , help='Used AMP?')
    arg('--subset', type=str, default='3Algorithms', help='the folder for three algorithms or nsf5?')
    arg('--device', type=str, default='cuda:0' , help='Device')
    arg("--test_single_image", help='test single image', action='store_true')
    arg('--batch_size', type=int, default=10, help='batch_size for test loader')
    arg('--num_workers', type=int, default=1, help='num-workers for test loader')

    
    args = parser.parse_args()
    #os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(':')[-1]
    #torch.cuda.set_device(int(args.device.split(':')[-1]))
    
    device = 'cuda:0'
    
    seed_everything(1994)
    os.makedirs(os.path.join(args.output,args.subset), exist_ok=True)
    if args.subset =='3Algorithms':
        net = get_net(args.model)
    else:
        net = get_net(args.model,out_features=2)
    # if using inplace_abn for speeding up, put args.surgery as 2

    # if args.surgery == 2:
    #     net = to_InPlaceABN(net)
    #     source = 'timm' if args.model.startswith('mixnet') else 'efficientnet-pytorch'
    #     net = to_MishME(net, source=source)
    if args.surgery == 1:
        source = 'timm' if args.model.startswith('mixnet') else 'efficientnet-pytorch'
        net = to_MishME(net, source=source)

    if args.subset =='3Algorithms':
        columns = [args.experiment + '_pc', args.experiment + '_pjm', args.experiment + '_pjuni', args.experiment + '_puerd']
        column_list = ['NAME', args.experiment+'_pc', args.experiment+'_pjm', args.experiment+'_pjuni', args.experiment+'_puerd']

    elif args.subset == 'for_nsf5':
        columns = [args.experiment + '_pc', args.experiment + '_pnsf5']
        column_list = ['NAME', args.experiment + '_pc', args.experiment + '_pnsf5']
    
    net = net.cuda(device)     
    checkpoint = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    # if args.fp16:
    #     net = amp.initialize(net, None, opt_level='O1',loss_scale='dynamic',verbosity=0)
    net.eval()

    args.test_time_augmentation = 1 if args.test_single_image else args.test_time_augmentation
    if args.test_time_augmentation == 4:
        TTA = dict()
        TTA['rot1'] =  lambda x: np.rot90(x,1)
        TTA['rot2'] =  lambda x: np.rot90(x,2)
        TTA['rot3'] =  lambda x: np.rot90(x,3)
        TTA['fliplr'] =  lambda x: np.fliplr(x)
        TTA['fliplr_rot1'] =  lambda x: np.rot90(np.fliplr(x),1)
        TTA['fliplr_rot2'] =  lambda x: np.rot90(np.fliplr(x),2)
        TTA['fliplr_rot3'] =  lambda x: np.rot90(np.fliplr(x),3)
        
    elif args.test_time_augmentation == 1:
        TTA = dict()
        TTA[''] = lambda x: x
    
    if not args.test_single_image:
        IL  = os.listdir(DATA_ROOT_PATH+args.folder)
        test_retriever = TestRetriever(IL, DATA_ROOT_PATH+args.folder, decoder=args.decoder)

        test_loader = torch.utils.data.DataLoader(
            test_retriever,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False)

        pred_dataframe = pd.DataFrame(columns=column_list)

        pred_dataframe['NAME'] = IL
        pred_dataframe[columns] = 0.0

        for transform in TTA.keys():
            y_preds = []
            test_loader.dataset.func_transforms = TTA[transform]

            for step, (image_names, images) in enumerate(tqdm(test_loader, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')):
                y_pred = net(images.cuda(device).float())
                y_pred = nn.functional.softmax(y_pred, dim=1).float().data.cpu().numpy()
                y_preds.extend(y_pred)

            y_preds = np.array(y_preds)

            pred_dataframe[columns] += y_preds


    else:
        file_name = args.folder.split("/")[-1]

        pred_dataframe = pd.DataFrame(
        columns=column_list)

        pred_dataframe['NAME'] = [file_name]
        pred_dataframe[columns] = 0.0
        if args.decoder == 'NR':
            tmp = jio.read(args.folder)
            image = decompress_structure(tmp).astype(np.float32)
            image = ycbcr2rgb(image)
            image /= 255.0
        else:
            image = cv2.imread(args.folder, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image /= 255.0

        transform = get_valid_transforms()
        transformed = transform(image=image)
        transformed_image = transformed['image']
        transformed_image = np.expand_dims(transformed_image, axis=0)
        for transform in TTA.keys():

            y_preds = []
            transformed_image = TTA[transform](transformed_image).copy()

            transformed_image = torch.from_numpy(transformed_image)

            y_pred = net(transformed_image.cuda(device).float())
            y_pred = nn.functional.softmax(y_pred, dim=1).float().data.cpu().numpy()

            y_preds.extend(y_pred)
            y_preds = np.array(y_preds)

            pred_dataframe[columns] += y_preds


    pred_dataframe[columns] /= len(TTA.keys())

    output_path = Path(os.path.join(args.output,args.subset,args.experiment+'_Test.csv'))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_dataframe.to_csv(output_path)
        
    del checkpoint
    del net


def pytorch_predict(folder ='/Test',model='efficientnet-b4',experiment='efficientnet_b4_NR_mish',surgery=0,test_time_augmentation=4,
                    checkpoint='',output='models_predictions/',decoder='R',fp16=0,subset='3Algorithms',device='cuda:0',test_single_image=False,batch_size=10,
                    num_workers=1):

    device = torch.device(device)
    DATA_ROOT_PATH = os.environ.get('DATA_ROOT_PATH')
    seed_everything(1994)
    os.makedirs(os.path.join(output, subset), exist_ok=True)
    if subset == '3Algorithms':
        net = get_net(model)
    else:
        net = get_net(model, out_features=2)
    # if using inplace_abn for speeding up, put args.surgery as 2

    # if args.surgery == 2:
    #     net = to_InPlaceABN(net)
    #     source = 'timm' if args.model.startswith('mixnet') else 'efficientnet-pytorch'
    #     net = to_MishME(net, source=source)
    if surgery == 1:
        source = 'timm' if model.startswith('mixnet') else 'efficientnet-pytorch'
        net = to_MishME(net, source=source)

    if subset == '3Algorithms':
        columns = [experiment + '_pc', experiment + '_pjm', experiment + '_pjuni',
                   experiment + '_puerd']
        column_list = ['NAME', experiment + '_pc', experiment + '_pjm', experiment + '_pjuni',
                       experiment + '_puerd']

    elif subset == 'for_nsf5':
        columns = [experiment + '_pc', experiment + '_pnsf5']
        column_list = ['NAME', experiment + '_pc', experiment + '_pnsf5']

    # net = net.cuda(device)
    net = net.to(device)
    checkpoint = torch.load(checkpoint, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    # if args.fp16:
    #     net = amp.initialize(net, None, opt_level='O1',loss_scale='dynamic',verbosity=0)
    net.eval()

    test_time_augmentation = 1 if test_single_image else test_time_augmentation
    if test_time_augmentation == 4:
        TTA = dict()
        TTA['rot1'] = lambda x: np.rot90(x, 1)
        TTA['rot2'] = lambda x: np.rot90(x, 2)
        TTA['rot3'] = lambda x: np.rot90(x, 3)
        TTA['fliplr'] = lambda x: np.fliplr(x)
        TTA['fliplr_rot1'] = lambda x: np.rot90(np.fliplr(x), 1)
        TTA['fliplr_rot2'] = lambda x: np.rot90(np.fliplr(x), 2)
        TTA['fliplr_rot3'] = lambda x: np.rot90(np.fliplr(x), 3)

    elif test_time_augmentation == 1:
        TTA = dict()
        TTA[''] = lambda x: x

    if not test_single_image:
        data_path = os.path.join(DATA_ROOT_PATH,folder)
        IL = os.listdir(data_path)
        test_retriever = TestRetriever(IL, data_path, decoder=decoder)

        test_loader = torch.utils.data.DataLoader(
            test_retriever,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False)

        pred_dataframe = pd.DataFrame(columns=column_list)

        pred_dataframe['NAME'] = IL
        pred_dataframe[columns] = 0.0

        for transform in TTA.keys():
            y_preds = []
            test_loader.dataset.func_transforms = TTA[transform]

            for step, (image_names, images) in enumerate(
                    tqdm(test_loader, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')):
                # y_pred = net(images.cuda(device).float())
                y_pred = net(images.to(device).float())
                y_pred = nn.functional.softmax(y_pred, dim=1).float().data.cpu().numpy()
                y_preds.extend(y_pred)

            y_preds = np.array(y_preds)

            pred_dataframe[columns] += y_preds


    else:
        file_name = folder.split("/")[-1]

        pred_dataframe = pd.DataFrame(
            columns=column_list)

        pred_dataframe['NAME'] = [file_name]
        pred_dataframe[columns] = 0.0
        if decoder == 'NR':
            tmp = jio.read(folder)
            image = decompress_structure(tmp).astype(np.float32)
            image = ycbcr2rgb(image)
            image /= 255.0
        else:
            image = cv2.imread(folder, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image /= 255.0

        transform = get_valid_transforms()
        transformed = transform(image=image)
        transformed_image = transformed['image']
        transformed_image = np.expand_dims(transformed_image, axis=0)
        for transform in TTA.keys():
            y_preds = []
            transformed_image = TTA[transform](transformed_image).copy()

            transformed_image = torch.from_numpy(transformed_image)

            # y_pred = net(transformed_image.cuda(device).float())
            y_pred = net(transformed_image.to(device).float())
            y_pred = nn.functional.softmax(y_pred, dim=1).float().data.cpu().numpy()

            y_preds.extend(y_pred)
            y_preds = np.array(y_preds)

            pred_dataframe[columns] += y_preds


    pred_dataframe[columns] /= len(TTA.keys())

    output_path = Path(os.path.join(output, subset, experiment + '_Test.csv'))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_dataframe.to_csv(output_path)
    print(experiment+'_Test.csv'+', saved done')
    del checkpoint
    del net
if __name__ == "__main__":
    main()