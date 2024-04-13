import argparse
import os
import random
import torch
import yaml
from collections import OrderedDict
from os import path as osp
import numpy as np


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        tuple: yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def yaml_load(f):
    """Load yaml file or string.

    Args:
        f (str): File path or a python string.

    Returns:
        dict: Loaded dict.
    """
    if os.path.isfile(f):
        with open(f, 'r') as f:
            return yaml.load(f, Loader=ordered_yaml()[0])
    else:
        return yaml.load(f, Loader=ordered_yaml()[0])


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg


def _postprocess_yml_value(value):
    # None
    if value == '~' or value.lower() == 'none':
        return None
    # bool
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    # !!float number
    if value.startswith('!!float'):
        return float(value.replace('!!float', ''))
    # number
    if value.isdigit():
        return int(value)
    elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:
        return float(value)
    # list
    if value.startswith('['):
        return eval(value)
    # str
    return value


def parse_options(root_path, is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')

    parser.add_argument(
        '--force_yml', nargs='+', default=None, help='Force to update yml files. Examples: train:ema_decay=0.999')
    args = parser.parse_args()

    # parse yml to dict
    opt = yaml_load(args.opt)

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = False

    # force to update yml options
    if args.force_yml is not None:
        for entry in args.force_yml:
            # now do not support creating new keys
            keys, value = entry.split('=')
            keys, value = keys.strip(), value.strip()
            value = _postprocess_yml_value(value)
            eval_str = 'opt'
            for key in keys.split(':'):
                eval_str += f'["{key}"]'
            eval_str += '=value'
            # using exec function
            exec(eval_str)

    if is_train:
        experiments_root = opt['path'].get('experiments_root')
        if experiments_root is None:
            experiments_root = osp.join(root_path, 'experiments')
        experiments_root = osp.join(experiments_root, opt['datasets']['type'], opt['datasets']['name'], opt['name'])

        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'saving_models')
        opt['path']['val'] = osp.join(experiments_root, 'fake_patches')
        opt['path']['log'] = osp.join(experiments_root, 'tb_logger')
    else:
        opt['path']['models'] = osp.join(root_path, 'experiments', opt['datasets']['type'], opt['datasets']['name'], opt['name'], 'saving_models')
        test_root = opt['path'].get('test_path')
        if test_root is None:
            test_root = osp.join(root_path, 'test_results')
        test_root = osp.join(test_root, opt['datasets']['type'], opt['datasets']['name'], opt['name'])
        opt['test_dir'] = test_root

    return opt, args


def copy_opt_file(opt_file, experiments_root):
    # copy the yml file to the experiment root
    import sys
    import time
    from shutil import copyfile
    cmd = ' '.join(sys.argv)
    filename = osp.join(experiments_root, osp.basename(opt_file))
    copyfile(opt_file, filename)

    with open(filename, 'r+') as f:
        lines = f.readlines()
        lines.insert(0, f'# GENERATE TIME: {time.asctime()}\n# CMD:\n# {cmd}\n\n')
        f.seek(0)
        f.writelines(lines)
