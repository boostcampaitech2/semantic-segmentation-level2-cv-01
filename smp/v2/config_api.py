import re
import sys
import os.path as osp
import tempfile
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
    def _substitute_predefined_vars(filename, temp_config_name):
        file_dirname = osp.dirname(filename)
        file_basename = osp.basename(filename)
        file_basename_no_extension = osp.splitext(file_basename)[0]
        file_extname = osp.splitext(filename)[1]
        support_templates = dict(
            fileDirname=file_dirname,
            fileBasename=file_basename,
            fileBasenameNoExtension=file_basename_no_extension,
            fileExtname=file_extname)
        with open(filename, 'r', encoding='utf-8') as f:
            # Setting encoding explicitly to resolve coding issue on windows
            config_file = f.read()
        for key, value in support_templates.items():
            regexp = r'\{\{\s*' + str(key) + r'\s*\}\}'
            value = value.replace('\\', '/')
            config_file = re.sub(regexp, value, config_file)
        with open(temp_config_name, 'w', encoding='utf-8') as tmp_config_file:
            tmp_config_file.write(config_file)

    @staticmethod
    def fromfile(file_name):
        file_name = osp.abspath(osp.expanduser(file_name))
        fileExtname = osp.splitext(file_name)[1]
        if not osp.isfile(file_name):
            raise FileNotFoundError(f"Config file '{file_name}' not found.")
        if fileExtname != '.py':
            raise IOError(f"Config file format '{fileExtname}' does not match '.py'.")

        with tempfile.TemporaryDirectory() as temp_config_dir:
            temp_config_file = tempfile.NamedTemporaryFile(
                dir=temp_config_dir, suffix=fileExtname)
            temp_config_name = osp.basename(temp_config_file.name)
            Config._substitute_predefined_vars(file_name,
                                                temp_config_file.name)
        
            temp_module_name = osp.splitext(temp_config_name)[0]
            sys.path.insert(0, temp_config_dir)

            mod = import_module(temp_module_name)
            temp_config_file.close()

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