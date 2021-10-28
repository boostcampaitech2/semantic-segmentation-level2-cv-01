import wandb

def make_wandb_images(images, outputs, masks, class_dict):
    """
    Log prediction result to wandb

    Args:
        images (obj : numpy array or tensor) : Original images

        outputs (obj : numpy array or tensor) : Model outputs (prediction results)

        masks (obj : numpy array or tensor) : Ground Truth

        class_dict (dict) : Class labels
            (keys: pixel values, values: string labels)

    Returns:
        result (obj : wandb image object) : Masked Images with ground truth
    """

    result = []
            
    for image, output, mask in zip(images, outputs, masks):
        result.append(wandb.Image(image, masks={
            "predictions" : {
                "mask_data" : output,
                "class_labels" : class_dict
                },
                "ground_truth" : {
                "mask_data" : mask,
                "class_labels" : class_dict
                }
            }
        ))
    
    return result

def log_lr(lr_list):
    """
    Log learning rate plot to wandb.

    Args:
        lr_list (list) : List of learning rates and steps in unit of epoch
            e.g. [[lr_1, setp_1], [lr_2, setp_2], ..., [lr_last, setp_last]]
    """
    if not lr_list:
        return
    table = wandb.Table(data=lr_list, columns=['Epoch', 'Learning rate'])
    wandb.log({'optimizer/lr' : wandb.plot.line(table, 'Epoch', 'Learning rate', title="optimizer/learning_rate")})

def log_train_wandb(metrics, epoch):
    log_dict = {'train/acc': metrics.mean_acc,
        'train/cls_acc': metrics.mean_mean_acc_cls,
        'train/mIoU': metrics.mean_mIoU,
        'train/loss': metrics.mean_loss,
        'train/fwavacc': metrics.mean_fwavacc}
        
    for i in range(metrics.n_class):
        log_dict[f'{metrics.classes[i]}/train_acc'] = metrics.mean_acc_each_cls[i]
        log_dict[f'{metrics.classes[i]}/train_IoU'] = metrics.mean_IoU[i]
    
    wandb.log(log_dict, step=epoch+1)

def log_valid_wandb(metrics, result_images, epoch):

    acc, acc_cls, mean_acc_cls, mIoU, fwavacc, IoU = metrics.label_accuracy_score()
    IoU_by_class = [{cls : round(IoU,4)} for IoU, cls in zip(IoU , metrics.classes)]

    log_dict = {'valid/acc': acc,
        'valid/cls_acc': mean_acc_cls,
        'valid/mIoU': mIoU,
        'valid/loss': metrics.mean_loss,
        'valid/fwavacc': fwavacc,
        'epoch/best_epoch':metrics.best_epoch}

    for i in range(metrics.n_class):
        log_dict[f'{metrics.classes[i]}/valid_acc'] = acc_cls[i]
        log_dict[f'{metrics.classes[i]}/valid_IoU'] = IoU[i]
    
    print(f'Validation #{epoch:2d}  Average Loss: {round(metrics.mean_loss, 4):7.4f}, Accuracy : {round(acc, 4):7.4f}, \
            mIoU: {round(mIoU, 4)}')
    print(f'IoU by class : {IoU_by_class}')
    
    wandb.log(log_dict, step=epoch)
    wandb.log({"validation":result_images}, step=epoch)