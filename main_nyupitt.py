"""
OCT retinal layer segmentation using U-Net based deep learning model.

This script implements training and testing functionality for segmenting retinal layers
from OCT B-scans acquired with Cirrus HD-OCT (Zeiss). The model segments 8 retinal layers:
RNFL, GCL+IPL, INL, OPL, ONL, IS, OS and RPE.

Key Features:
- Training and validation on OCT image datasets
- Testing with or without ground truth segmentation masks
- Calculation of Dice scores and pixel accuracy metrics
- Visualization of segmentation results
- Support for both CPU and GPU processing

Example Usage:
    See Readme.md for detailed instructions on running the script.

The script expects data organized in specific folder structure:
    data_dir/
        train/
            img/    - Training images
            mask/   - Training segmentation masks
        eval/
            img/    - Validation images  
            mask/   - Validation masks
        test/
            img/    - Test images
            mask/   - Test masks
        predict/     (OPTIONAL)
            img/    - Images to predict on (no masks required)
"""
N_CLASSES = 9  # including background

import os
import os.path as osp
import argparse

import logging
import time
import copy

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import data.seg_transforms as dt
from data.seg_dataset import segList
from utils.logger import Logger
from models.nets.OSNet import OSMGUNet
from utils.loss import loss_builder2
from utils.utils import adjust_learning_rate
from utils.utils import AverageMeter,save_model
from utils.utils import compute_dice,compute_pa,compute_average_score_for_one_image
from utils.vis import vis_result

# logger vis
FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger_vis = logging.getLogger(__name__)
logger_vis.setLevel(logging.DEBUG)

def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print("NEW FOLDER:", os.path.basename(dir_path))
    return dir_path

# training process
def train(args,train_loader, model, criterion2, optimizer,epoch,print_freq=10):
    """Train model for one epoch.

    Performs training on batches of OCT images, computing loss and Dice scores
    for the segmentation output. Updates model weights using backpropagation.

    Args:
        args: Arguments containing training parameters 
        train_loader: DataLoader for training data
        model: Neural network model
        criterion2: Loss function
        optimizer: Optimization algorithm
        epoch: Current epoch number
        print_freq: Print metrics every N iterations

    Returns:
        tuple containing:
            - Average loss for the epoch
            - Average Dice scores (overall and per layer)
    """
    # trains for one batch - it is used in "train_seg" function.
    # set the AverageMeter
    batch_time = AverageMeter()
    losses = AverageMeter()
    dice = AverageMeter()
    Dice_1, Dice_2, Dice_3, Dice_4 = AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter()
    Dice_5, Dice_6, Dice_7, Dice_8 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # variable
        input_var = Variable(input).to(args.device)
        target_var_seg = Variable(target).to(args.device)
        # target_var_seg = Variable(target).long().to(args.device)
        input_var1 = copy.deepcopy(input_var)
        # forward
        output_seg = model(input_var1)

        # calculate loss
        loss_2_1 = criterion2[0](output_seg, target_var_seg)
        loss_2_2 = criterion2[1](output_seg, target_var_seg)
        loss_2= loss_2_1 + loss_2_2     # loss from the two-stage network       
        loss = loss_2
        losses.update(loss.data, input.size(0))
        # calculate dice score for segmentation 
        _, pred_seg = torch.max(output_seg, 1)
        pred_seg = pred_seg.cpu().data.numpy()
        label_seg = target_var_seg.cpu().data.numpy()

        ret_d = compute_dice(label_seg, pred_seg, n_classes=N_CLASSES)
        # dice_score = compute_single_avg_score(ret_d)
        dice_score = compute_average_score_for_one_image(ret_d)

        # update dice score
        dice.update(dice_score)
        Dice_1.update(ret_d[1])
        Dice_2.update(ret_d[2])
        Dice_3.update(ret_d[3])
        Dice_4.update(ret_d[4])
        Dice_5.update(ret_d[5])
        Dice_6.update(ret_d[6])
        Dice_7.update(ret_d[7])
        Dice_8.update(ret_d[8])

        # backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # logger vis
        if i % print_freq == 0:
            logger_vis.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Dice {dice.val:.4f} ({dice.avg:.4f})\t'
                        'Dice_1 {dice_1.val:.4f} ({dice_1.avg:.4f})\t'
                        'Dice_2 {dice_2.val:.4f} ({dice_2.avg:.4f})\t'
                        'Dice_3 {dice_3.val:.4f} ({dice_3.avg:.4f})\t'
                        'Dice_4 {dice_4.val:.4f} ({dice_4.avg:.4f})\t'
                        'Dice_5 {dice_5.val:.4f} ({dice_5.avg:.4f})\t'
                        'Dice_6 {dice_6.val:.4f} ({dice_6.avg:.4f})\t'
                        'Dice_7 {dice_7.val:.4f} ({dice_7.avg:.4f})\t'
                        'Dice_8 {dice_8.val:.4f} ({dice_8.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,dice = dice,dice_1=Dice_1,dice_2=Dice_2,dice_3=Dice_3,dice_4=Dice_4,dice_5=Dice_5,dice_6=Dice_6,dice_7=Dice_7,dice_8=Dice_8))
            print('Loss :',loss.cpu().data.numpy())
    return losses.avg,dice.avg,Dice_1.avg,Dice_2.avg,Dice_3.avg,Dice_4.avg,Dice_5.avg,Dice_6.avg,Dice_7.avg,Dice_8.avg

# evaluation process
def eval_predict(phase, args, eval_data_loader, model, result_path = None):
    """Run prediction on dataset without ground truth masks.

    Performs inference using trained model and saves visualization results.
    Used when only input images are available without segmentation masks.

    Args:
        phase: Current phase ('predict')
        args: Runtime arguments
        eval_data_loader: DataLoader for test images
        model: Trained neural network model
        result_path: Path to save visualization results
    """
    batch_time = AverageMeter()
    model.eval()
    end = time.time()
    for iter, (image, imt, imn) in enumerate(eval_data_loader):
        with torch.no_grad():
            image_var = Variable(image).to(args.device)
            # model forward
            # _, _, output_seg = model(image_var)
            output_seg = model(image_var)
            _, pred_seg = torch.max(output_seg, 1)

            # save visualized result
            pred_seg = pred_seg.cpu().data.numpy().astype('uint8')
            batch_time.update(time.time() - end)
            end = time.time()

            save_dir = osp.join(result_path, 'vis')
            # if not exists(save_dir + '/pred'): os.makedirs(save_dir + '/pred')
            create_directory(save_dir + '/pred')
            imn = os.path.basename(imn[0])
            imt = (imt.squeeze().numpy()).astype('uint8')
            vis_result(imn, imt, None, pred_seg, save_dir)
            print('Saved visualized results!')
            
def eval(phase, args, eval_data_loader, model,result_path = None, logger = None):
    """Evaluate model performance on validation/test set.

    Computes metrics including Dice scores and pixel accuracy for each retinal layer.
    Saves visualization results and logs metrics.

    Args:
        phase: Current phase ('train'/'eval'/'test') 
        args: Runtime arguments
        eval_data_loader: DataLoader for evaluation
        model: Neural network model
        result_path: Path to save results
        logger: Logger to record metrics

    Returns:
        tuple containing:
            - Final Dice scores (overall and per layer)
            - List of Dice scores for each image
    """
    # set the AverageMeter 
    batch_time = AverageMeter()
    dice = AverageMeter()
    mpa = AverageMeter()

    Dice_1, Dice_2, Dice_3, Dice_4 = AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter()
    Dice_5, Dice_6, Dice_7, Dice_8 = AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter()
    pa_1, pa_2, pa_3, pa_4 = AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter()
    pa_5, pa_6, pa_7, pa_8 = AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter()

    dice_list, mpa_list = [], []
    ret_dice, ret_pa = [], []
    # switch to eval mode
    model.eval()
    end = time.time()
    pred_seg_batch = []
    label_seg_batch = []
    for iter, (image, label, imt, imn) in enumerate(eval_data_loader):
        with torch.no_grad():
            image_var = Variable(image).to(args.device)
            # model forward
            output_seg = model(image_var)
            _, pred_seg = torch.max(output_seg, 1)
            # save visualized result
            pred_seg = pred_seg.cpu().data.numpy().astype('uint8')
            if phase == 'eval' or phase == 'test':
                imt = (imt.squeeze().numpy()).astype('uint8')
                ant = label.numpy().astype('uint8')
                save_dir = osp.join(result_path, 'vis')
                # if not exists(save_dir): os.makedirs(save_dir)
                create_directory(save_dir)
                # if not exists(save_dir+'/label'):os.makedirs(save_dir+'/label')
                create_directory(save_dir + '/label')
                # if not exists(save_dir + '/pred'): os.makedirs(save_dir + '/pred')
                create_directory(save_dir + '/pred')
                imn = os.path.basename(imn[0])                              
                vis_result(imn, imt, ant, pred_seg, save_dir)
                print('Saved visualized results!')
            # calculate dice and pa score for segmentation
            label_seg = label.numpy().astype('uint8')
            pred_seg_batch.append(pred_seg)
            label_seg_batch.append(label_seg)
            ret_d = compute_dice(label_seg, pred_seg, n_classes=N_CLASSES)
            ret_p = compute_pa(label_seg, pred_seg, n_classes=N_CLASSES)
            ret_dice.append(ret_d)
            ret_pa.append(ret_p)

            # dice_score = compute_single_avg_score(ret_d)
            # mpa_score = compute_single_avg_score(ret_p)

            dice_score = compute_average_score_for_one_image(ret_d)
            mpa_score = compute_average_score_for_one_image(ret_p)

            dice_list.append(dice_score)
            # update dice and pa score
            dice.update(dice_score)
            Dice_1.update(ret_d[1])
            Dice_2.update(ret_d[2])
            Dice_3.update(ret_d[3])
            Dice_4.update(ret_d[4])
            Dice_5.update(ret_d[5])
            Dice_6.update(ret_d[6])
            Dice_7.update(ret_d[7])
            Dice_8.update(ret_d[8])
            mpa_list.append(mpa_score)
            mpa.update(mpa_score)
            pa_1.update(ret_p[1])
            pa_2.update(ret_p[2])
            pa_3.update(ret_p[3])
            pa_4.update(ret_p[4])
            pa_5.update(ret_p[5])
            pa_6.update(ret_p[6])
            pa_7.update(ret_p[7])
            pa_8.update(ret_p[8])
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger_vis.info('{0}: [{1}/{2}]\t'
                        'ID {id}\t'
                        'Dice {dice.val:.4f}\t'
                        'Dice_1 {dice_1.val:.4f}\t'
                        'Dice_2 {dice_2.val:.4f}\t'
                        'Dice_3 {dice_3.val:.4f}\t'
                        'Dice_4 {dice_4.val:.4f}\t'
                        'Dice_5 {dice_5.val:.4f}\t'
                        'Dice_6 {dice_6.val:.4f}\t'
                        'Dice_7 {dice_7.val:.4f}\t'
                        'Dice_8 {dice_8.val:.4f}\t'
                        'MPA {mpa.val:.4f}\t'
                        'PA_1 {pa_1.val:.4f}\t'
                        'PA_2 {pa_2.val:.4f}\t'
                        'PA_3 {pa_3.val:.4f}\t'
                        'PA_4 {pa_4.val:.4f}\t'
                        'PA_5 {pa_5.val:.4f}\t'
                        'PA_6 {pa_6.val:.4f}\t'
                        'PA_7 {pa_7.val:.4f}\t'
                        'PA_8 {pa_8.val:.4f}\t'
                        'Batch_time {batch_time.val:.3f}\t'
                        .format(phase.upper(), iter, len(eval_data_loader),id=imn[0].split('.')[0], dice=dice, dice_1=Dice_1, dice_2=Dice_2, dice_3=Dice_3,
                                dice_4=Dice_4, dice_5=Dice_5, dice_6=Dice_6, dice_7=Dice_7, dice_8=Dice_8,
                                mpa=mpa, pa_1=pa_1, pa_2=pa_2, pa_3=pa_3,
                                pa_4=pa_4, pa_5=pa_5, pa_6=pa_6, pa_7=pa_7, pa_8=pa_8,
                                batch_time=batch_time))
    # print final all dice and pa score 
    final_dice_avg, final_dice_1, final_dice_2, final_dice_3, final_dice_4, final_dice_5, final_dice_6, final_dice_7, final_dice_8 = dice.avg, Dice_1.avg, Dice_2.avg, Dice_3.avg, Dice_4.avg, Dice_5.avg, Dice_6.avg, Dice_7.avg, Dice_8.avg
    final_pa_avg, final_pa_1, final_pa_2, final_pa_3, final_pa_4, final_pa_5, final_pa_6, final_pa_7, final_pa_8 = mpa.avg, pa_1.avg, pa_2.avg, pa_3.avg, pa_4.avg, pa_5.avg, pa_6.avg, pa_7.avg, pa_8.avg
    print('######  Segmentation Result  ######')
    print('Final Dice_avg Score:{:.4f}'.format(final_dice_avg))
    print('Final Dice_1 Score:{:.4f}'.format(final_dice_1))
    print('Final Dice_2 Score:{:.4f}'.format(final_dice_2))
    print('Final Dice_3 Score:{:.4f}'.format(final_dice_3))
    print('Final Dice_4 Score:{:.4f}'.format(final_dice_4))
    print('Final Dice_5 Score:{:.4f}'.format(final_dice_5))
    print('Final Dice_6 Score:{:.4f}'.format(final_dice_6))
    print('Final Dice_7 Score:{:.4f}'.format(final_dice_7))
    print('Final Dice_8 Score:{:.4f}'.format(final_dice_8))

    print('Final PA_avg:{:.4f}'.format(final_pa_avg))
    print('Final PA_1 Score:{:.4f}'.format(final_pa_1))
    print('Final PA_2 Score:{:.4f}'.format(final_pa_2))
    print('Final PA_3 Score:{:.4f}'.format(final_pa_3))
    print('Final PA_4 Score:{:.4f}'.format(final_pa_4))
    print('Final PA_5 Score:{:.4f}'.format(final_pa_5))
    print('Final PA_6 Score:{:.4f}'.format(final_pa_6))
    print('Final PA_7 Score:{:.4f}'.format(final_pa_7))
    print('Final PA_8 Score:{:.4f}'.format(final_pa_8))

    if phase == 'eval' or phase == 'test':
        logger.append(
        [ final_dice_avg, final_dice_1, final_dice_2, final_dice_3, final_dice_4, final_dice_5, final_dice_6, final_dice_7, final_dice_8,
        final_pa_avg, final_pa_1, final_pa_2, final_pa_3, final_pa_4, final_pa_5, final_pa_6, final_pa_7, final_pa_8])
    return final_dice_avg, final_dice_1, final_dice_2, final_dice_3, final_dice_4, final_dice_5, final_dice_6, final_dice_7, final_dice_8,dice_list


###### train ######
def train_seg(args,train_result_path,train_loader,eval_loader):
    """Main training function.

    Handles the complete training process including:
    - Setting up loggers and metrics tracking
    - Initializing model, loss and optimizer
    - Training for specified epochs
    - Periodic evaluation on validation set
    - Model checkpointing

    Args:
        args: Training arguments/parameters
        train_result_path: Path to save training results
        train_loader: DataLoader for training data
        eval_loader: DataLoader for validation data
    """
    # logger setting
    logger_train = Logger(osp.join(train_result_path,'dice_epoch.txt'), title='dice',resume=False)
    metric_list = ['Epoch', 'Dice_Train', 'Dice_Val']
    for ii in range(1, N_CLASSES):
        metric_list.append(f'Dice_{ii}')      # to store current value
        metric_list.append(f'Dice_{ii}{ii}')  # to store average value
    logger_train.set_names(metric_list)
    # logger_train.set_names(['Epoch','Dice_Train','Dice_Val',
    #                         'Dice_1','Dice_11','Dice_2','Dice_22','Dice_3','Dice_33','Dice_4','Dice_44',
    #                         'Dice_5','Dice_55','Dice_6','Dice_66','Dice_7','Dice_77',
    #                         'Dice_8','Dice_88'
    #                         ])

    # print hyperparameters
    for k, v in args.__dict__.items():
        print(k, ':', v)
    # load the network
    net = OSMGUNet()
    model = torch.nn.DataParallel(net).to(args.device)

    # define loss function
    criterion2 = loss_builder2(class_num=N_CLASSES,
                               ignore_index=-100)
    # set optimizer
    optimizer = torch.optim.Adam(net.parameters(), #Adam optimizer
                                    args.lr,
                                    betas=(0.9, 0.99),
                                    weight_decay=args.weight_decay)     
    cudnn.benchmark = True

    # main training
    print('#' * 15, 'TRAINING', '#' * 15)
    best_dice = 0
    start_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(args,optimizer, epoch)
        logger_vis.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))
        # train for one epoch
        loss,dice_train,dice_1,dice_2,dice_3,dice_4,dice_5,dice_6,dice_7,dice_8 = train(args,train_loader,
                                                                                        model,criterion2, optimizer,epoch,
                                                                                        print_freq=100)
        # evaluate on validation set
        dice_val,dice_11,dice_22,dice_33,dice_44,\
            dice_55,dice_66,dice_77,dice_88,dice_list = eval('train', args, eval_loader, model)
        # save the best model
        is_best = dice_val > best_dice
        best_dice = max(dice_val, best_dice)
        model_dir = osp.join(train_result_path,'model')
        # if not exists(model_dir):
        #     os.makedirs(model_dir)
        create_directory(model_dir)
        save_model({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'dice_epoch':dice_val,
            'best_dice': best_dice,
        }, is_best, model_dir)
        # logger 
        logger_train.append([epoch,dice_train,dice_val,dice_1,dice_11,dice_2,
                             dice_22,dice_3,dice_33,dice_4,dice_44,dice_5,dice_55,
                             dice_6,dice_66,dice_7,dice_77,dice_8,dice_88,])


###### test ######
def test_seg(args, test_result_path, test_loader):
    """Test trained model on test dataset.

    Loads trained model and evaluates performance on test set.
    Can handle both scenarios:
    - Testing with ground truth masks (computing metrics)
    - Prediction only (without ground truth)

    Args:
        args: Testing arguments including model path
        test_result_path: Path to save test results
        test_loader: DataLoader for test data
    """

    # load the model
    print('Loading test model ...')
    net = OSMGUNet()
    model = torch.nn.DataParallel(net).to(args.device)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint['state_dict'])
    print('Model loaded!')
    cudnn.benchmark = True
    # test the model on test set
    # eval('test', args, test_loader, model, test_result_path, logger_test)

    phase = test_loader.dataset.phase
    if phase == "predict":
        eval_predict(phase, args, test_loader, model, test_result_path)
    else:
        logger_test = Logger(osp.join(test_result_path, 'dice_mpa_epoch.txt'), title='dice&mpa', resume=False)
        logger_test.set_names(
            ['Dice', 'Dice_1', 'Dice_2', 'Dice_3', 'Dice_4',
             'Dice_5', 'Dice_6', 'Dice_7', 'Dice_8',
             'mpa', 'pa_1', 'pa_2', 'pa_3', 'pa_4',
             'pa_5', 'pa_6', 'pa_7', 'pa_8' ])
        eval(phase, args, test_loader, model, test_result_path, logger_test)

def parse_args():
    parser = argparse.ArgumentParser(description='train')
    # config
    parser.add_argument('-d', '--data-dir', default=None, required=True)
    parser.add_argument('--log_path', default=None, required=True)
    parser.add_argument('--local-test', dest='local_test', action='store_true')
    parser.add_argument('--predict', dest='predict_phase', action='store_true')
    parser.add_argument('-j', '--workers', type=int, default=1)
    # train setting
    parser.add_argument('--step', type=int, default=20,
                        help="Show metric values when `step` number of samples are presented during each epoch")
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-mode', type=str, default='step')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--test-name', type=str, default='test1',
                        help="a suffix added to result (or log) path")
    parser.add_argument('--model-path',
                        help='Pretrained model path - If it is not empty, no training happens',
                        default="", type=str)
    args = parser.parse_args()
    if args.log_path is None:
        args.log_path = "logs/OCTseg-Zeiss/"
    return args

def main():
    
    # region CONFIG                   
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    args.__dict__["device"] = device

    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print('torch version:',torch.__version__)

    # path setting
    tn = args.test_name                 # a suffix that will be added to the output folder's name
    dataset_name = args.data_dir.split('/')[-1]

    # path_prefix = "d:/" if args.local_test else "../"
    # args.data_dir = path_prefix + args.data_dir
    # args.log_path = path_prefix + args.log_path
    # args.model_path = path_prefix + args.model_path if args.model_path != "" else ""

    print('#' * 15, "IMPORTANT PATHS", '#' * 15)
    print('data_dir:',args.data_dir)
    print('log_path:',args.log_path)
    print('model_path:',args.model_path)
    # endregion
    
    # region LOAD DATASET

    if dataset_name.lower().find("onh") >-1:
        PIXEL_MEAN = [0.14162449977018857]*3
        PIXEL_STD = [0.09798050174952816]*3
    else:
        raise Exception("PIXEL_MEAN AND PIXEL_STD WERE NOT DEFINED")
    normalize = dt.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)
    
    t = []
    if args.predict_phase == True:
        # When there is no ground truth
        test_phase = "predict"
        t.extend([dt.ToTensor(), normalize])
    else:
        test_phase = "test"
        t.extend([dt.Label_Transform_NYUPITT(), dt.ToTensor(), normalize])
        # t.extend([dt.Label_Transform_NYUPITT_v2(), dt.ToTensor(), normalize])
        
    if args.model_path == "":
        train_dataset = segList(args.data_dir, 'train', dt.Compose(t))
        val_dataset = segList(args.data_dir, 'eval', dt.Compose(t))

        train_result_path = osp.join(args.log_path,"OCTseg-Zeiss",dataset_name,'train', tn + '_' +str(args.lr))
        # if not exists(train_result_path):
        #     os.makedirs(train_result_path)
        create_directory(train_result_path)

    test_dataset = segList(args.data_dir, test_phase, dt.Compose(t))
    test_result_path = osp.join(args.log_path, "OCTseg-Zeiss", dataset_name, test_phase, tn + '_' +str(args.lr))

    # if not exists(test_result_path):
    #     os.makedirs(test_result_path)
    create_directory(test_result_path)
    print("TEST RESULT PATH:", test_result_path)

    if args.model_path == "":
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,
                                                   shuffle=True, num_workers=args.workers,
                                                   pin_memory=True, drop_last=True)
        eval_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                  shuffle=False, num_workers=args.workers,
                                                  pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                              num_workers=args.workers, pin_memory=False)
    # endregion
    
    # region TRAIN-VALID-TEST
    if args.model_path == "":

        train_seg(args,train_result_path,train_loader,eval_loader)

        print('#' * 15, "TEST PRETRAINED MODEL", '#' * 15)
        model_best_path = osp.join(osp.join(train_result_path,'model'),'model_best.pth.tar')
        args.model_path = model_best_path
        test_seg(args,test_result_path,test_loader)
    else:
        test_seg(args, test_result_path, test_loader)
    # endregion

if __name__ == '__main__':
    main()
