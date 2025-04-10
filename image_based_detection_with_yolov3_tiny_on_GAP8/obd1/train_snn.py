# nosec # noqa
import os
import argparse
from typing import Any, Dict, Tuple
import cv2
import numpy as np
from PIL import Image
#import matplotlib
#matplotlib.use('Agg')
#from matplotlib import pyplot as plt
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
#from models import yolo_base,tiny_yolov3_str
from obd1 import dataset,models
from obd1.boundingbox import metrics,utils
from obd1.models import ann_new_model_2,yolo_base
from obd1.dataset import evCIVIL
from spikingjelly.activation_based import functional

from lava.lib.dl import slayer
#from lava.lib.dl.slayer import obd


#change the number of input channels to 1
#change the input resolution (240,320) - (h,w)
#change the letterbox to letterbox2 in evCIVIL
#change the feeding values to vflip to 240 and hflip to 320
#uncommneted rotate90 stuff
#change the make_dvs_frame from color to gray to facilitate for 1 channel.
#init_model change to (240,320) and 1 channel.
#change the bbox scaling in _getitem_ method according to 240,320

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=int, default=[0], help='which gpu(s) to use', nargs='+')
    parser.add_argument('-b',   type=int, default=16,  help='batch size for dataloader')
    parser.add_argument('-verbose', default=False, action='store_true', help='lots of debug printouts')
    # Model
    parser.add_argument('-model', type=str, default='tiny_yolov3_snn', help='network model')
    # Sparsity
    parser.add_argument('-sparsity', action='store_true', default=False, help='enable sparsity loss')
    parser.add_argument('-sp_lam',   type=float, default=0.01, help='sparsity loss mixture ratio')
    parser.add_argument('-sp_rate',  type=float, default=0.01, help='minimum rate for sparsity penalization')
    # Optimizer
    parser.add_argument('-lr',  type=float, default=0.001, help='initial learning rate')
    parser.add_argument('-wd',  type=float, default=1e-5,   help='optimizer weight decay')
    parser.add_argument('-lrf', type=float, default=0.01,   help='learning rate reduction factor for lr scheduler')
    # Network/SDNN
    parser.add_argument('-threshold',  type=float, default=1.0, help='neuron threshold')
    parser.add_argument('-tau_grad',   type=float, default=0.1, help='surrogate gradient time constant')
    parser.add_argument('-scale_grad', type=float, default=0.2, help='surrogate gradient scale')
    parser.add_argument('-clip',       type=float, default=10, help='gradient clipping limit')
    # Pretrained model
    #parser.add_argument('-load', type=str, default='', help='pretrained model')
    parser.add_argument('-load', type=str, default='/work3/kniud/object_detection/SNN/spikingjelly/yolov3/yolov3_ann_best_model_2_head1/Trained_sigmoid_tiny_yolov3_snn/epoch_447_0.3544013564001861.pt', help='pretrained model')

    # Target generation
    parser.add_argument('-tgt_iou_thr', type=float, default=0.5, help='ignore iou threshold in target generation')
    # YOLO loss
    parser.add_argument('-lambda_coord',    type=float, default=1.0, help='YOLO coordinate loss lambda')
    parser.add_argument('-lambda_noobj',    type=float, default=2.0, help='YOLO no-object loss lambda')
    parser.add_argument('-lambda_obj',      type=float, default=2.0, help='YOLO object loss lambda')
    parser.add_argument('-lambda_cls',      type=float, default=4.0, help='YOLO class loss lambda')
    parser.add_argument('-lambda_iou',      type=float, default=2.0, help='YOLO iou loss lambda')
    parser.add_argument('-alpha_iou',       type=float, default=0.8, help='YOLO loss object target iou mixture factor')
    parser.add_argument('-label_smoothing', type=float, default=0.1, help='YOLO class cross entropy label smoothing')
    parser.add_argument('-track_iter',      type=int,  default=1000, help='YOLO loss tracking interval')
    # Experiment
    parser.add_argument('-exp',  type=str, default='',   help='experiment differentiater string')
    parser.add_argument('-seed', type=int, default=None, help='random seed of the experiment')
    # Training
    parser.add_argument('-epoch',  type=int, default=600, help='number of epochs to run')
    parser.add_argument('-warmup', type=int, default=10,  help='number of epochs to warmup')
    # dataset
    parser.add_argument('-dataset',     type=str,   default='evCIVIL', help='dataset to use [BDD100K]')
    parser.add_argument('-path',        type=str,   default='/home/udayanga/latest_dataset/', help='dataset path')
    parser.add_argument('-output_dir',  type=str,   default='/home/udayanga/Udaya_Research_stuff/2024_GAP8_work/yolov3_ann_model_images/out', help='directory in which to put log folders')
    parser.add_argument('-num_workers', type=int,   default=8, help='number of dataloader workers')
    parser.add_argument('-aug_prob',    type=float, default=0.2, help='training augmentation probability')
    parser.add_argument('-clamp_max',   type=float, default=5.0, help='exponential clamp in height/width calculation')
    parser.add_argument("-train_csv_file",type=str,default="night_outdoor_and_daytime_train_files_image_based.txt",help="csv file...")
    parser.add_argument("-test_csv_file",type=str,default="test_files_image_based.txt",help="csv file...")
    parser.add_argument("-conf_thres",type=float,default="0.2")
    parser.add_argument("--tbins",type=int,default=1,help="")
    parser.add_argument("--TSteps",type=int,default=4,help="")
    parser.add_argument("--optimizer",type = str, default="SGD",help="")
    parser.add_argument("--scheduler",type = str, default="cosine",help="")

    args = parser.parse_args()

    identifier = f'{args.model}_' + args.exp if len(args.exp) > 0 else args.model
    if args.seed is not None:
        torch.manual_seed(args.seed)
        identifier += '_{}'.format(args.seed)

    trained_folder = args.output_dir + '/Trained_sigmoid_' + \
        identifier if len(identifier) > 0 else args.output_dir + '/Trained'
    logs_folder = args.output_dir + '/Logs_' + \
        identifier if len(identifier) > 0 else args.output_dir + '/Logs_sigmoid_'
    
    print(trained_folder)
    writer = SummaryWriter(args.output_dir + '/runs/' + identifier)

    os.makedirs(trained_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)

    with open(trained_folder + '/args.txt', 'wt') as f:
        for arg, value in sorted(vars(args).items()):
            f.write('{} : {}\n'.format(arg, value))
    
    print('Using GPUs {}'.format(args.gpu))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes_output = {'evCIVIL': 2}
    print('Creating Network')
    #udayanga : change channels to 1
    net = ann_new_model_2.Network(in_channels=1,threshold=args.threshold,
                      tau_grad=args.tau_grad,
                      scale_grad=args.scale_grad,
                      num_classes=2,
                      clamp_max=args.clamp_max).to(device)

    functional.reset_net(net)
    functional.set_step_mode(net, step_mode='m')
    
    net.initialize_model_weights()
    #udayanga : channel change
    net.init_model((320,320),1)     #2 * args.tbins)

    net.to(device)

    print('Creating Optimizer')
    """optimizer = torch.optim.Adam(net.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.wd)

    # Define learning rate s-heduler
    def lf(x):
        return (min(x / args.warmup, 1)
                * ((1 + np.cos(x * np.pi / args.epoch)) / 2)
                * (1 - args.lrf)
                + args.lrf)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)"""

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr,weight_decay=5e-4)
    """optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay = 0.0005,
                                nesterov = True)"""
    
    yolo_target = yolo_base.YOLOtarget(anchors= net.anchors,
                                 scales= net.scale,
                                 num_classes=net.num_classes,
                                 ignore_iou_thres=args.tgt_iou_thr)

    print('Creating Dataset')

    param_dict = {"TSteps" : args.TSteps, "tbins" : args.tbins ,"quantized_h" : 260 ,"quantized_w" : 346}

    train_set = evCIVIL.evCIVIL(root = args.path ,csv_file_name = args.train_csv_file , param_dict = param_dict, train = True,augment = True)
    test_set = evCIVIL.evCIVIL(root = args.path, csv_file_name= args.test_csv_file, param_dict = param_dict, train = False, augment = False)

    train_loader = DataLoader(train_set,
                                batch_size=args.b,
                                shuffle=True,
                                collate_fn=yolo_target.collate_fn,
                                num_workers=args.num_workers,
                                pin_memory=True)
    
    test_loader = DataLoader(test_set,
                                batch_size=8,
                                shuffle=False,
                                collate_fn=yolo_target.collate_fn,    
                                num_workers=args.num_workers,
                                pin_memory=True)
    
    box_color_map = [(np.random.randint(256),
                          np.random.randint(256),
                          np.random.randint(256))
                         for i in range(11)]
    

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            args.epoch,        # * len(train_loader),
    )
    
    print('Creating YOLO Loss')
    yolo_loss = models.yolo_base.YOLOLoss(anchors=net.anchors,
                             lambda_coord=args.lambda_coord,
                             lambda_noobj=args.lambda_noobj,
                             lambda_obj=args.lambda_obj,
                             lambda_cls=args.lambda_cls,
                             lambda_iou=args.lambda_iou,
                             alpha_iou=args.alpha_iou,
                             label_smoothing=args.label_smoothing).to(device)
    
    print('Creating Stats Module')
    stats = slayer.utils.LearningStats(accuracy_str='AP@0.5')

    loss_tracker = dict(coord=[], obj=[], noobj=[], cls=[], iou=[])
    loss_order = ['coord', 'obj', 'noobj', 'cls', 'iou']

    current_epoch = 0

    if args.sparsity:
        sparsity_montior = slayer.loss.SparsityEnforcer(
            max_rate=args.sp_rate, lam=args.sp_lam)
    else:
        sparsity_montior = None

    print('Training/Testing Loop')

    pretrained_model_path = args.load

    """pretrained_model_info = torch.load(pretrained_model_path,map_location=device)
    model_state_dict = pretrained_model_info["model_state_dict"]
    net.load_state_dict(model_state_dict)
    optimizer_dict = pretrained_model_info["optimizer"]
    optimizer.load_state_dict(optimizer_dict)
    scheduler_dict = pretrained_model_info["scheduler"]
    scheduler.load_state_dict(scheduler_dict)
    current_epoch = pretrained_model_info["epoch"]
    print("loading state dictionary ")"""

    current_val_acc = 0.
    for epoch in range(current_epoch,args.epoch):
        t_st = datetime.now()
        ap_stats = metrics.APstats(iou_threshold=0.5)

        print(f'{epoch=}')
        net.train()
        
        for i, (inputs, targets, bboxes) in enumerate(train_loader):

                inputs = inputs.permute(4,0,1,2,3)
          
                inputs = inputs.squeeze(0)
     
                inputs = inputs.to(device)
                
                print('forward') if args.verbose else None
                predictions, counts = net(inputs, sparsity_montior)

                
                predictions = [prediction.unsqueeze(-1) for prediction in predictions]
            
                loss, loss_distr = yolo_loss(predictions, targets)

                if sparsity_montior is not None:
                    loss += sparsity_montior.loss
                    sparsity_montior.clear()

                if torch.isnan(loss):
                    print("loss is nan, continuing")
                    continue

                optimizer.zero_grad()
                loss.backward()
                net.validate_gradients()
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
                optimizer.step()
                
        
                # MAP calculations
                T = 1 #inputs.shape[-1]
                try:
                    predictions = torch.concat([net.yolo(p, a) for (p, a)
                                                in zip(predictions, net.anchors)],
                                            dim=1)
                except RuntimeError:
                    print('Runtime error on MAP predictions calculation.'
                        'continuing')
                    continue

                predictions = [utils.nms(predictions[..., t],conf_threshold = args.conf_thres)
                            for t in range(T)]
                
                for t in range(T):
                    ap_stats.update(predictions[t], bboxes[t])

                if not torch.isnan(loss):
                    stats.training.loss_sum += loss.item() * inputs.shape[0]
                stats.training.num_samples += inputs.shape[0]
                stats.training.correct_samples = ap_stats[:] * \
                    stats.training.num_samples

                processed = i * train_loader.batch_size
                total = len(train_loader.dataset)
                time_elapsed = (datetime.now() - t_st).total_seconds()
                samples_sec = time_elapsed / (i + 1) / train_loader.batch_size
                header_list = [f'Train: [{processed}/{total} '
                            f'({100.0 * processed / total:.0f}%)]']
                header_list += ['Event Rate: ['
                                + ', '.join([f'{c.item():.2f}'
                                            for c in counts[0]]) + ']']
                header_list += [f'Coord loss: {loss_distr[0].item()}']
                header_list += [f'Obj   loss: {loss_distr[1].item()}']
                header_list += [f'NoObj loss: {loss_distr[2].item()}']
                header_list += [f'Class loss: {loss_distr[3].item()}']
                header_list += [f'IOU   loss: {loss_distr[4].item()}']

                if i % args.track_iter == 0:
                    #plt.figure()
                    for loss_idx, loss_key in enumerate(loss_order):
                        loss_tracker[loss_key].append(loss_distr[loss_idx].item())
                        #plt.semilogy(loss_tracker[loss_key], label=loss_key)
                        writer.add_scalar(f'Loss Tracker/{loss_key}',
                                            loss_distr[loss_idx].item(),
                                            len(loss_tracker[loss_key]) - 1)
                        
                stats.print(epoch, i, samples_sec, header=header_list)
                #functional.reset_net(net)
        
        t_st = datetime.now()
        ap_stats = metrics.APstats(iou_threshold=0.5)
        net.eval()
        
        with torch.no_grad():
            for i, (inputs, targets, bboxes) in enumerate(test_loader):
                
                """"inputs = inputs.permute(4,0,1,2,3)
                inputs = inputs.to(device)
                predictions, counts = net(inputs)

                T = 1
                predictions = [utils.nms(predictions[..., t],conf_threshold = args.conf_thres)
                    for t in range(T)]
                for t in range(T):
                    ap_stats.update(predictions[t], bboxes[t])"""

                #sta
                inputs = inputs.permute(4,0,1,2,3)
          
                inputs = inputs.squeeze(0)

                inputs = inputs.to(device)
                predictions, counts = net(inputs, sparsity_montior)
      

                predictions = [prediction.unsqueeze(-1) for prediction in predictions]
                
                #predictions = [torch.sum(prediction,dim=-1).unsqueeze(-1) for prediction in predictions]

                # MAP calculations
                T = 1 #inputs.shape[-1]
                try:
                    predictions = torch.concat([net.yolo(p, a) for (p, a)
                        in zip(predictions, net.anchors)],dim=1)
                except RuntimeError:
                    print('Runtime error on MAP predictions calculation.'
                            'continuing')
                    continue

                predictions = [utils.nms(predictions[..., t],conf_threshold = args.conf_thres)
                                for t in range(T)]
                for t in range(T):
                    ap_stats.update(predictions[t], bboxes[t])
                    #end

                #stats.testing.loss_sum += loss.item() * inputs.shape[0]
                stats.testing.num_samples += inputs.shape[0]
                stats.testing.correct_samples = ap_stats[:] * stats.testing.num_samples

                processed = i * test_loader.batch_size
                total = len(test_loader.dataset)
                time_elapsed = (datetime.now() - t_st).total_seconds()
                samples_sec = time_elapsed / (i + 1) / test_loader.batch_size
                header_list = [f'Test: [{processed}/{total} '
                                    f'({100.0 * processed / total:.0f}%)]']
                """header_list += ['Event Rate: ['
                                        + ', '.join([f'{c.item():.2f}'
                                                    for c in counts[0]]) + ']']
                header_list += [f'Coord loss: {loss_distr[0].item()}']
                header_list += [f'Obj   loss: {loss_distr[1].item()}']
                header_list += [f'NoObj loss: {loss_distr[2].item()}']
                header_list += [f'Class loss: {loss_distr[3].item()}']
                header_list += [f'IOU   loss: {loss_distr[4].item()}']
                stats.print(epoch, i, samples_sec, header=header_list)"""
                #functional.reset_net(net)

        #writer.add_scalar('Loss/train', stats.training.loss, epoch)
        #writer.add_scalar('mAP@50/train', stats.training.accuracy, epoch)
        writer.add_scalar('mAP@50/test', stats.testing.accuracy, epoch)

        print("dumping checkpoint ",stats.testing.accuracy)

        if stats.testing.accuracy > current_val_acc:
            checkpoint = {"epoch": epoch,
                            "model_state_dict": net.state_dict(),   #module.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict()}
            full_ckpt_path = trained_folder + "/epoch_" + str(epoch) + "_" + str(stats.testing.accuracy) + ".pt"
            print("saving ")
            torch.save(checkpoint,full_ckpt_path)
            current_val_acc = stats.testing.accuracy
        else:
            if epoch % 2 == 0:
                    checkpoint = {"epoch": epoch,
                                "model_state_dict": net.state_dict(),   #module.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict()}
                    full_ckpt_path = trained_folder + "/epoch_" + str(epoch) + "_" + str(stats.testing.accuracy) + ".pt"
                    print("saving ")
                    torch.save(checkpoint,full_ckpt_path)
        
        stats.update()
        stats.save(trained_folder + '/')
        scheduler.step()
