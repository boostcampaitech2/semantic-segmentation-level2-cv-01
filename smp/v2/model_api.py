import os
import torch
import segmentation_models_pytorch as smp

def build_model(cfg_model):
    if hasattr(smp, cfg_model.type):
        _model = getattr(smp, cfg_model.type)
    else:
        raise KeyError(f"smp library has no model named '{cfg_model.type}'.")
    model = _model(**cfg_model.args)
    return model
    
def build_module(parent_module, cfg_module, has_attr, **kwargs):
    if not has_attr:
        return None
    if hasattr(parent_module, cfg_module.type):
        module = getattr(parent_module, cfg_module.type)
        if hasattr(cfg_module, 'args'):
            module = module(**cfg_module.args, **kwargs)
        else:
            module = module(**kwargs)
    else:
        raise KeyError(f"{parent_module} has no module named '{cfg_module.type}'.")

    return module

def save_model(model, saved_dir, file_name, debug=False):
    """
    Save model in state_dict format.

    Args :
        model (obj : torch.nn.Module) : Model to use

        saved_dir (str) : Directory path where to save model

        file_name (str) : Name of model to be saved

        debug (bool) : Debugging mode (default : False)
    """
    if debug:
        return

    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(check_point, output_path)