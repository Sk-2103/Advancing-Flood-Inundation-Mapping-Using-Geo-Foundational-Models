import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs
from medpy.metric.binary import asd, hd
import os
import csv
def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    logger, 
                    config, 
                    scaler=None):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        images, targets, file_name = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
        if config.amp:
            with autocast():
                out = model(images)
                loss = criterion(out, targets)      
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step() 


def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk, file_name = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        print(confusion)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)
    
    return np.mean(loss_list)


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    hd_list = []
    asd_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk,file_name = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)
            # Convert predictions and ground truth to binary masks
            bin_pred = np.where(out >= config.threshold, 1, 0)
            bin_gt = np.where(msk >= 0.5, 1, 0)

            # Compute Hausdorff Distance (HD) and Average Surface Distance (ASD)
            try:
                hausdorff_dist = hd(bin_pred, bin_gt, voxelspacing=None)
                avg_surface_dist = asd(bin_pred, bin_gt, voxelspacing=None)
            except Exception as e:
                hausdorff_dist, avg_surface_dist = np.nan, np.nan
                print(f'Error in computing HD or ASD: {str(e)}')

            hd_list.append(hausdorff_dist)
            asd_list.append(avg_surface_dist)
            save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name, file_name=file_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        avg_hd = np.nanmean(hd_list) if len(hd_list) > 0 else np.nan
        avg_asd = np.nanmean(asd_list) if len(asd_list) > 0 else np.nan

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}, HD: {avg_hd:.4f}, ASD: {avg_asd:.4f}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)


def test_all_images(test_loader, model, criterion, logger, config, csv_folder, test_data_name=None):
    model.eval()
    results = []
    loss_list = []
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk, file_name = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            
            msk = msk.squeeze(1).cpu().detach().numpy()
            out = out.squeeze(1).cpu().detach().numpy()
            
            bin_pred = np.where(out >= config.threshold, 1, 0)
            bin_gt = np.where(msk >= 0.5, 1, 0)
            
            y_pre = bin_pred.flatten()
            y_true = bin_gt.flatten()
            
            confusion = confusion_matrix(y_true, y_pre, labels=[0, 1])
            TN, FP, FN, TP = confusion.ravel() if confusion.size == 4 else (0, 0, 0, 0)
            
            accuracy = (TN + TP) / np.sum(confusion) if np.sum(confusion) != 0 else 0
            sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
            f1_or_dsc = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0
            miou = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0
            
            results.append([file_name[0], loss.item(), miou, f1_or_dsc, accuracy, specificity, sensitivity])
            
    csv_path = os.path.join(csv_folder, 'test_results.csv')
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['File Name', 'Loss', 'mIoU', 'F1/DSC', 'Accuracy', 'Specificity', 'Sensitivity'])
        writer.writerows(results)
    
    log_info = f'Test results saved to {csv_path}'
    print(log_info)
    logger.info(log_info)
    
    return np.mean(loss_list)


