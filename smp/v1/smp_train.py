import argparse
import numpy as np
import random
import torch
import os
import torch.nn as nn
import wandb

import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import label_accuracy_score, add_hist
from custom import CustomDataset, collate_fn

def parse_args():
    """
    Parse arguments from terminal. Use Args in terminel when execute
    this script file.
    e.g. python smp_Unet2plus.py --debug --deterministic --private

    Args:
        optional:
            debug : Activate debugging mode
                (default : False)
            
            deterministic : Set seed for reproducibility
                (default : False)
            
            private : Log result in private wandb entity
                (default : False)
    """
    parser = argparse.ArgumentParser(description='Train Segmentation Model')
    parser.add_argument('--debug',
        action='store_true',
        help='whether to use small dataset for debugging')

    parser.add_argument('--deterministic',
        action='store_true',
        help='whether to set random seed for reproducibility')

    parser.add_argument('--private',
        action='store_true',
        help='whether to log in private wandb entity')
    
    args = parser.parse_args()

    return args

def make_dataloader(mode='train', batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn, debug=False):
    """
    Create dataloader by arguments.

    Args:
        mode (str) : Type of dataset (default : 'train')
            e.g. mode='train', mode='val', mode='test'
        
        batch_size (int) : Batch size (default : 8)

        suffle (bool) : Whether to shuffle dataset when creating loader
            (default : False)
        
        num_workers (int) : Number of processors (default : 4)
        
        collate_fn (func) : Collate function for Dataset
            (default : collate_fn from custom)
        
        debug (bool) : Debugging mode (default : False)

    Returns:
        loader (obj : DataLoader) : DataLoader created by arguments
    """
    annotation = {'train':'train.json',
                'val':'val.json',
                'test':'test.json'}

    train_transform = A.Compose([
                            ToTensorV2()
                            ])

    val_transform = A.Compose([
                            ToTensorV2()
                            ])

    test_transform = A.Compose([
                            ToTensorV2()
                            ])

    transforms = {'train':train_transform,
                'val':val_transform,
                'test':test_transform}

    dataset = CustomDataset(annotation=annotation[mode], mode=mode, transform=transforms[mode])
    if debug:
        dataset = dataset.split_dataset(ratio=0.1)
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        collate_fn=collate_fn)
    
    return loader

def set_random_seed(random_seed=21):
    """
    Set seed.

    Args:
        random_seed (int) : Seed to be set (default : 21)
    """
    random_seed = 21
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def make_save_dir(saved_dir):
    """
    Make Directory to save checkpoints. This function has been added
    to avoid saving different experiments of same models in one directory.
    So you can prevent some of checkpoints from being changed.

    Args :
        saved_dir (str) : Path of directory to save checkpoints.

    Returns :
        saved_dir (str) : New directory path
    """

    while os.path.isdir(saved_dir):
        components = saved_dir.split('_')
        if len(components) > 1 and 'exp' in components[-1]:
            exp_str = components[-1]
            exp_num = int(exp_str[3:])
            saved_dir = saved_dir[:-len(str(exp_num))] + str(exp_num + 1)
        else:
            saved_dir += '_exp2'

    if not os.path.isdir(saved_dir):                                                           
        os.mkdir(saved_dir)

    return saved_dir

def save_model(model, saved_dir, file_name, debug=False):
    """
    Save model in state_dict format.

    Args :
        model (obj : torch.nn.Module) : Model to use

        saved_dir (str) : Directory path where to save model

        file_name (str) : Name of model to be saved

        debug (bool) : Debugging mode (default : False)
    """
    # if debug:
    #     return

    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(check_point, output_path)

def train(num_epochs, model, train_loader, val_loader, criterion, optimizer, saved_dir, val_every, device, debug):
    """
    Train segmentation model.

    Args:
        num_epochs (int) : Number of Epochs

        model (obj : torch.nn.Module) : Model to train

        train_loader (obj : DataLoader) : Loader for model train

        val_loader (obj : DataLoader) : Loader for model validation

        criterion (obj : torch.nn.Loss) : Loss function

        optimizer (obj : torch.optim.Optimizer) : Optimizer

        saved_dir (str) : Directory path where to save model

        val_every (int) : Validation interval

        device (str) : Processor ('cuda' or 'cpu')
    """
    print(f'Start training..')
    n_class = 11
    best_loss = 9999999
    best_mIoU = -1
    cats = ['Backgroud',
            'General trash',
            'Paper',
            'Paper pack',
            'Metal',
            'Glass',
            'Plastic',
            'Styrofoam',
            'Plastic bag',
            'Battery',
            'Clothing']
    
    for epoch in range(num_epochs):
        model.train()
        mean_acc = 0
        mean_acc_each_cls = np.zeros(n_class)
        mean_mean_acc_cls = 0
        mean_mIoU = 0
        mean_loss = 0
        mean_fwavacc = 0
        mean_IoU = np.zeros(n_class)

        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(train_loader):
            images = torch.stack(images)       
            masks = torch.stack(masks).long() 
            
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)
            
            # device 할당
            model = model.to(device)
            
            # inference
            outputs = model(images)
            
            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, acc_cls, mean_acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)

            mean_acc += acc
            mean_acc_each_cls += np.array(acc_cls)
            mean_mean_acc_cls += mean_acc_cls
            mean_mIoU += mIoU
            mean_loss += loss.item()
            mean_fwavacc += fwavacc
            mean_IoU += np.array(IoU)
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], \
                        Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')

        # calculate metric
        mean_acc /= len(train_loader)
        mean_acc_each_cls /= len(train_loader)
        mean_mean_acc_cls /= len(train_loader)
        mean_mIoU /= len(train_loader)
        mean_loss /= len(train_loader)
        mean_fwavacc /= len(train_loader)
        mean_IoU /= len(train_loader)

        log_dict = {'train/acc': mean_acc,
                'train/cls_acc': mean_mean_acc_cls,
                'train/mIoU': mean_mIoU,
                'train/loss': mean_loss,
                'train/fwavacc': mean_fwavacc}
                
        for i in range(n_class):
            log_dict[f'{cats[i]}/train_acc'] = mean_acc_each_cls[i]
            log_dict[f'{cats[i]}/train_IoU'] = mean_IoU[i]
        
        wandb.log(log_dict, step=epoch+1)
             
        # validation 주기에 따른 loss 출력 및 best model 저장
        ## TODO
        save_interval = 3
        if (epoch + 1) % val_every == 0:
            avrg_loss, val_mIoU = validation(epoch + 1, model, val_loader, criterion, device)
            # if avrg_loss < best_loss and best_mIoU < val_mIoU:
            if best_mIoU < val_mIoU: 
                print(f"Best Performance at epoch: {epoch + 1}")
                best_mIoU = val_mIoU
                save_model(model, saved_dir, file_name=f'{model.name}_best.pt', debug=debug)
                if (epoch + 1) % save_interval == 0:
                    save_model(model, saved_dir, file_name=f'{model.name}_{epoch+1}.pt', debug=debug)
    save_model(model, saved_dir, file_name=f'{model.name}_last.pt', debug=debug)

def validation(epoch, model, data_loader, criterion, device):
    """
    Validate segmentation model.

    Args:
        epoch (int) : Current Epoch (start from 1)

        model (obj : torch.nn.Module) : Model to validate

        data_loader (obj : DataLoader) : Loader for model validation

        criterion (obj : torch.nn.Loss) : Loss function

        device (str) : Processor ('cuda' or 'cpu')

    Returns:
        avrg_loss (float) : Average Loss of validation
        
        mIoU (float) : mean IoU of validation for every class
    """
    print(f'Start validation #{epoch}')
    model.eval()
    cats = ['Backgroud',
            'General trash',
            'Paper',
            'Paper pack',
            'Metal',
            'Glass',
            'Plastic',
            'Styrofoam',
            'Plastic bag',
            'Battery',
            'Clothing']

    with torch.no_grad():
        n_class = 11
        total_loss = 0
        cnt = 0
        
        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(data_loader):
            
            images = torch.stack(images)       
            masks = torch.stack(masks).long()  

            images, masks = images.to(device), masks.to(device)            
            
            # device 할당
            model = model.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
        
        acc, acc_cls, mean_acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , cats)]
        avrg_loss = total_loss / cnt

        log_dict = {'valid/acc': acc,
            'valid/cls_acc': mean_acc_cls,
            'valid/mIoU': mIoU,
            'valid/loss': avrg_loss.item(),
            'valid/fwavacc': fwavacc}

        for i in range(n_class):
            log_dict[f'{cats[i]}/valid_acc'] = acc_cls[i]
            log_dict[f'{cats[i]}/valid_IoU'] = IoU[i]
        
        wandb.log(log_dict, step=epoch)

        print(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, \
                mIoU: {round(mIoU, 4)}')
        print(f'IoU by class : {IoU_by_class}')
        
    return avrg_loss, mIoU

if __name__ == '__main__':
    # print init settings
    print("="*30 + '\n')
    print('pytorch version: {}'.format(torch.__version__))
    print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))

    print(torch.cuda.get_device_name(0))
    print(torch.cuda.device_count())

    args = parse_args()
    debug = args.debug
    deterministic = args.deterministic
    private = args.private

    print(f"Debugging Mode : {debug}")
    print(f"Deterministic : {deterministic}")
    print(f"Private wandb mode : {private}")
    print('\n' + "="*30 + '\n')

    if deterministic:
        set_random_seed()

    # GPU 사용 가능 여부에 따라 device 정보 저장
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define model
    ## TODO
    model = smp.Linknet(
        encoder_name='tu-hrnet_w64',
        in_channels=3,
        classes=11
    )

    # Hyperparameter 정의
    ## TODO
    val_every = 1
    batch_size = 16   # Mini-batch size
    num_epochs = 50
    learning_rate = 0.0001
    saved_dir = os.path.join('/opt/ml/segmentation/saved', model.name)
    saved_dir = make_save_dir(saved_dir)

    # wandb
    ## TODO
    if private:
        entity_name = 'bagineer'
    else:
        entity_name = 'perforated_line'

    # DataLoader 정의
    ## TODO
    train_loader = make_dataloader(mode='train',
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=4,
                                collate_fn=collate_fn,
                                debug=debug)

    val_loader = make_dataloader(mode='val',
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=collate_fn,
                                debug=debug)

    # Loss function 정의
    ## TODO
    criterion = nn.CrossEntropyLoss()

    # Optimizer 정의
    ## TODO
    optimizer = torch.optim.Adam(params = model.parameters(), lr = learning_rate, weight_decay=1e-6)

    wandb_config = {
        'val_every': val_every,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'Loss': criterion.__class__.__name__,
        'Optimizer': optimizer.__class__.__name__,
        'learning_rate': learning_rate
    }

    wandb.init(project='smp', entity=entity_name, config=wandb_config)
    run_name = model.name
    if debug:
        run_name = 'debug_' + run_name        
    wandb.run.name = run_name
    wandb.run.save()

    # Start training
    train(num_epochs, model, train_loader, val_loader, criterion, optimizer, saved_dir, val_every, device, debug)