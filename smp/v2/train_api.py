import torch
from tqdm import tqdm

from init_api import set_exp_name
from model_api import build_module, save_model
from metric_api import Metrics
from wandb_api import log_lr, make_wandb_images, log_train_wandb, log_valid_wandb

def train_model(model, train_loader, valid_loader, saved_dir, cfg, debug=False):
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

    print('Start Training..')

    model_name = set_exp_name(cfg, debug)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = build_module(torch.nn, cfg.criterion, hasattr(cfg, 'criterion'))
    optimizer = build_module(torch.optim, cfg.optimizer, hasattr(cfg, 'optimizer'), params=model.parameters())
    scheduler = build_module(torch.optim.lr_scheduler, cfg.scheduler, hasattr(cfg, 'optimizer'), optimizer=optimizer)

    best_loss = 9999999
    steps = 0
    lr_list = []
    
    epoch_pbar = tqdm(range(cfg.num_epochs),
                        total=cfg.num_epochs,
                        ncols=100,
                        position=1,
                        leave=True)

    for epoch in epoch_pbar:
        model.train()
        metrics = Metrics(cfg.classes, len(train_loader))

        train_pbar = tqdm(enumerate(train_loader),
                            total=len(train_loader),
                            ncols=50,
                            position=0,
                            leave=False)

        for step, (images, masks, _) in train_pbar:
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
            if scheduler:
                scheduler.step()
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()

            metrics.add_hist(masks, outputs)
            metrics.accumulate(loss)
            steps += 1

            if scheduler:
                lr_list.append([steps/metrics.len_loader, scheduler.get_last_lr()[0]])
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                desc_str = [f'Epoch [{epoch+1:2d}/{cfg.num_epochs:2d}]',
                            f'Step [{step+1:4d}/{metrics.len_loader:4d}]',
                            f'Loss: {round(metrics.loss,4):7.4f}',
                            f'mIoU: {round(metrics.mIoU,4):7.4f}']
                print(desc_str)
        
        # calculate metric
        metrics.update()
        log_train_wandb(metrics, epoch)
             
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % cfg.val_every == 0:
            avrg_loss, val_mIoU = validation(epoch + 1, model, valid_loader, criterion, device, cfg)
            # if avrg_loss < best_loss and best_mIoU < val_mIoU:
            if metrics.best_mIoU < val_mIoU: 
                print(f"Best Performance at epoch: {epoch + 1:2d}")
                metrics.best_mIoU = val_mIoU
                metrics.bset_epoch = epoch + 1
                save_model(model, saved_dir, file_name=f'{model_name}_best.pt', debug=debug)
                if (epoch + 1) % cfg.save_interval == 0:
                    save_model(model, saved_dir, file_name=f'{model_name}_{epoch+1}.pt', debug=debug)

    log_lr(lr_list)
    save_model(model, saved_dir, file_name=f'{model_name}_last.pt', debug=debug)

def validation(epoch, model, valid_loader, criterion, device, cfg):
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
    print(f'Start validation #{epoch:2d}')
    model.eval()
    metrics = Metrics(cfg.classes, len(valid_loader))

    with torch.no_grad():
        result_images = []

        valid_pbar = tqdm(enumerate(valid_loader),
                            total=len(valid_loader),
                            ncols=50,
                            position=0,
                            leave=False)

        for step, (images, masks, _) in valid_pbar:
            
            images = torch.stack(images)       
            masks = torch.stack(masks).long()  

            images, masks = images.to(device), masks.to(device)            
            
            # device 할당
            model = model.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()

            metrics.add_hist(masks, outputs)
            metrics.accumulate_loss(loss)
            result_images += make_wandb_images(images, outputs, masks, metrics.classes)
        
        metrics.update_loss()
        log_valid_wandb(metrics, result_images, epoch)
        
    return metrics.mean_loss, metrics.mIoU