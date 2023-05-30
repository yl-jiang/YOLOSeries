import logging
from tabulate import tabulate
import pprint

# __name__: 调用logger.py文件的那个脚本的名称
logger = logging.getLogger(__name__)

__all__ = ["print_config", "assemble_hyp"]


def assemble_hyp(hyp_dict):
    def fill_one_grid(k, v, key_max_len, val_max_len):
        grid_str = '├' + "─" * key_max_len + "┼" + '─' * val_max_len + "┤" + "\n"
        grid_str +=  '│' + ' ' + k + ' ' * (key_max_len - len(k) - 1) + '│' +  ' ' + v + " " * (val_max_len - len(v) - 1) + "│" + '\n'
        return grid_str
    key_max_len = max([len(str(k)) for k in hyp_dict.keys()]) + 2
    val_max_len = max([len(str(v)) for v in hyp_dict.values()]) + 2
    print_str = "\n"
    for i, (k, v) in enumerate(hyp_dict.items()):
        if i == 0: # title
            print_str += "╒" + "═" * key_max_len + "╤" + "═" * val_max_len + "╕" + '\n'
            print_str += '│' + ' ' + "KEYS" + ' ' * (key_max_len - len("KEYS") - 1) + '│' +  ' ' + "VALUES" + " " * (val_max_len - len("VALUES") - 1) + "│" + '\n'
            print_str += '╞' + '═' * key_max_len + '╪' + "═" * val_max_len + '╡' +  '\n'
            print_str += '│' + ' ' + str(k) + ' ' * (key_max_len - len(str(k)) - 1) + '│' +  ' ' + str(v) + " " * (val_max_len - len(str(v)) - 1) + "│" + '\n'
        else:
            print_str += fill_one_grid(str(k), str(v), key_max_len, val_max_len)
    print_str += '╘' + "═" * key_max_len + "╧" + '═' * val_max_len + "╛" + "\n"
    return str(print_str)


def print_config(args):
    table_header = ["keys", "values"]
    if isinstance(args, dict):
        exp_table = [
            (str(k), pprint.pformat(v))
            for k, v in args.items()
            if not k.startswith("_")
        ]
    else:
        exp_table = [
            (str(k), pprint.pformat(v))
            for k, v in vars(args).items()
            if not k.startswith("_")
        ]
    return tabulate(exp_table, headers=table_header, tablefmt="fancy_grid")    

if __name__ == "__main__":
    hyp = {"a": 'xxxxx', 'b': 1, 'cssssss': 'vvvvvvvvv', 'd': False}
    print(assemble_hyp(hyp))
