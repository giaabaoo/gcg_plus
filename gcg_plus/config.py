import argparse
import os
import re

import yaml


def setup_config():
    """Parse stacked YAML config files and optional CLI overrides.

    Usage:
        python main.py --config configs/base.yml configs/models/llama3_8b.yml configs/datasets/strongreject.yml

    CLI overrides (dot-notation, type-annotated):
        --attack.num_steps 500:int
        --model.temperature 0.0:float
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', nargs='+', required=True)
    opt, unknown_args = parser.parse_known_args()

    configs = [Config(path) for path in opt.config]
    config = merge_configs(*configs)
    config.parse_args(unknown_args)
    config.parse_refs()

    config.output_dir = os.path.join(config.output_dir, config.exp_name)
    return config


class Config(dict):
    """Dict subclass with dot-access and @{ref} cross-reference support."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        for arg in args:
            if arg == ' ':
                continue
            if isinstance(arg, str) and arg.endswith('.yml'):
                with open(arg, 'r') as f:
                    _init_assign(self, yaml.safe_load(f))
            elif isinstance(arg, dict):
                _init_assign(self, arg)
            else:
                raise TypeError(f'Expected .yml path or dict, got {type(arg)}')
        if kwargs:
            _init_assign(self, kwargs)

    def __call__(self, *args, **kwargs):
        return Config(self, *args, **kwargs)

    def __repr__(self, indent=4, prefix=''):
        lines = []
        for key, value in sorted(self.items()):
            if isinstance(value, Config):
                lines.append(f'{prefix}{key}:')
                lines.append(value.__repr__(indent, prefix + ' ' * indent))
            else:
                lines.append(f'{prefix}{key}: {value}')
        return '\n'.join(lines)

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]

    def __getitem__(self, key):
        sub_cfg, sub_key = _consume_dots(self, key, create_default=False)
        return dict.__getitem__(sub_cfg, sub_key)

    def __setitem__(self, key, value):
        sub_cfg, sub_key = _consume_dots(self, key, create_default=True)
        if sub_cfg.__contains__(sub_key) and value == '_DELETE_CONFIG_':
            dict.__delitem__(sub_cfg, sub_key)
        else:
            dict.__setitem__(sub_cfg, sub_key, value)

    def __delitem__(self, key):
        sub_cfg, sub_key = _consume_dots(self, key, create_default=False)
        dict.__delitem__(sub_cfg, sub_key)

    def __contains__(self, key):
        try:
            _consume_dots(self, key, create_default=False)
        except KeyError:
            return False
        return True

    def all_items(self, only_leaf=True):
        yield from _traverse_dfs(self, 'item', only_leaf=only_leaf)

    def parse_args(self, cmd_args):
        index = 0
        while index < len(cmd_args):
            arg = cmd_args[index]
            assert arg.startswith('--') and len(arg) > 2
            arg = arg[2:]
            if '=' in arg:
                key, full_value_str = arg.split('=', 1)
                index += 1
            else:
                key = arg
                full_value_str = cmd_args[index + 1]
                index += 2

            if ':' in full_value_str:
                value_str, type_str = full_value_str.split(':', 1)
                value_type = eval(type_str)
            else:
                value_str = full_value_str
                value_type = None

            if key not in self:
                raise KeyError(f'Unknown config key: {key}')

            if value_type is None:
                value_type = type(self[key])
            if value_type is bool:
                self[key] = value_str.lower() in ('true', '1', 'yes')
            else:
                self[key] = value_type(value_str)

    def parse_refs(self, subconf=None, _depth=1):
        if _depth > 10:
            raise RuntimeError('Circular @{ref} detected in config')
        if subconf is None:
            subconf = self
        for key in subconf.keys():
            value = subconf[key]
            if isinstance(value, str) and '@' in value:
                if value.startswith('@{') and value.endswith('}') and value.count('@') == 1:
                    subconf[key] = self[value[2:-1]]
                else:
                    for ref_key in set(re.findall(r"'@\{(.+?)\}'", value)):
                        value = value.replace(f"'@{{{ref_key}}}'", str(self[ref_key]))
                    subconf[key] = value
        for key in subconf.keys():
            if isinstance(subconf[key], Config):
                self.parse_refs(subconf[key], _depth + 1)


def merge_configs(*configs):
    result = Config()
    for cfg in configs:
        result = result(dict(cfg.all_items()))
    return result


# ── internal helpers ──────────────────────────────────────────────────────────

def _consume_dots(config, key, create_default):
    parts = key.split('.', 1)
    sub_key = parts[0]
    if sub_key in Config.__dict__:
        raise KeyError(f'"{sub_key}" is a reserved Config method name')
    if not dict.__contains__(config, sub_key) and len(parts) == 2:
        if create_default:
            dict.__setitem__(config, sub_key, Config())
        else:
            raise KeyError(key)
    if len(parts) == 1:
        return config, sub_key
    sub = dict.__getitem__(config, sub_key)
    if not isinstance(sub, Config):
        if create_default:
            sub = Config()
            dict.__setitem__(config, sub_key, sub)
        else:
            raise KeyError(key)
    return _consume_dots(sub, parts[1], create_default)


def _traverse_dfs(root, mode, only_leaf=True, prefix=''):
    for key, value in root.items():
        full_key = f'{prefix}.{key}'.lstrip('.')
        children = list(_traverse_dfs(value, mode, only_leaf, full_key)) if isinstance(value, Config) else []
        if not (children and only_leaf):
            yield {'key': full_key, 'value': value, 'item': (full_key, value)}[mode]
        yield from children


def _init_assign(config, d, prefix=''):
    for key, value in d.items():
        full_key = f'{prefix}.{key}'.lstrip('.') if prefix else key
        if isinstance(value, dict):
            _init_assign(config, value, full_key)
        else:
            sub_cfg, sub_key = _consume_dots(config, full_key, create_default=True)
            dict.__setitem__(sub_cfg, sub_key, value)
