import os.path as osp
import easydict
from importlib import import_module

class Config:

    NECESSARY = ['model', 'optimizer', 'criterion', 'data', 'train_pipeline', 'test_pipeline']
        
    @staticmethod
    def list_to_easy(arg_list):
        easy_list = []
        for arg in arg_list:
            if isinstance(arg, dict):
                easy_list.append(easydict.EasyDict(arg))
            elif isinstance(arg, str):
                return arg_list
            elif isinstance(arg, int):
                return arg_list
            elif isinstance(arg, float):
                return arg_list
            else:
                raise TypeError(f"{arg} is not type of '(dict / str / int / float)'.")
        return easy_list

    @staticmethod
    def mod_to_easydict(mod):
        dictionary = dict()
        for key, value in mod.__dict__.items():
            if isinstance(value, list):
                dictionary[key] = Config.list_to_easy(value)
            else:
                if not key.startswith('__'):
                    dictionary[key] = value
        return easydict.EasyDict(dictionary)

    @staticmethod
    def fromfile(file_name):
        file_name = osp.abspath(osp.expanduser(file_name))
        fileExtname = osp.splitext(file_name)[1]
        if not osp.isfile(file_name):
            raise FileNotFoundError(f"Config file '{file_name}' not found.")
        if fileExtname != '.py':
            raise IOError(f"Config file format '{fileExtname}' does not match '.py'.")

        base_name = osp.basename(file_name)
        module_name = osp.splitext(base_name)[0]
        mod = import_module(module_name)

        cfg = Config.mod_to_easydict(mod)
        
        return Config(cfg)

    def __init__(self, *cfg, **kwargs):

        for dictionary in cfg:
            subset = set(Config.NECESSARY).difference(set(dictionary.keys()))
            if subset:
                raise KeyError(f"'{subset}'' is not defined in your config file.")
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])