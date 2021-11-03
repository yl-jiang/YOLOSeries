import yaml
from pathlib import Path

cwd = Path("__file__").absolute().parent

class Config:
    def __init__(self) -> None:
        self.config = {}

    def update_config(self, args):
        if hasattr(args, 'batch_size'):
            self.config.update({'batch_size': args.batch_size})
        if hasattr(args, 'input_img_size'):
            self.config.update({'input_img_size': [args.input_img_size, args.input_img_size]})
        if hasattr(args, 'use_focal_loss'):
            self.config.update({'use_focal_loss': args.use_focal_loss})
        if hasattr(args, 'img_dir'):
            self.config.update({'img_dir': args.img_dir})
        if hasattr(args, 'lab_dir'):
            self.config.update({'lab_dir': args.lab_dir})
        if hasattr(args, 'name_path'):
            self.config.update({'name_path': args.name_path})
        if hasattr(args, 'cache_num'):
            self.config.update({'cache_num': args.cache_num})
        if hasattr(args,'model_save_dir'):
            self.config.update({'model_save_dir': args.model_save_dir})
        if hasattr(args,'log_save_path'):
            self.config.update({'log_save_path': args.log_save_path})
        if hasattr(args,'total_epoch'):
            self.config.update({'total_epoch': args.total_epoch})
        if hasattr(args,'do_warmup'):
            self.config.update({'do_warmup': args.do_warmup})
        if hasattr(args,'optimizer'):
            self.config.update({'optimizer': args.optimizer})
        if hasattr(args,'iou_threshold'):
            self.config.update({'iou_threshold': args.iou_threshold})
        if hasattr(args,'conf_threshold'):
            self.config.update({'conf_threshold': args.conf_threshold})
        if hasattr(args,'cls_threshold'):
            self.config.update({'cls_threshold': args.cls_threshold})
        if hasattr(args,'agnostic'):
            self.config.update({'agnostic': args.agnostic})
        if hasattr(args,'init_lr'):
            self.config.update({'init_lr': args.init_lr})
        if hasattr(args,'pretrained_model_path'):
            self.config.update({"pretrained_model_path": args.pretrained_model_path})
        if hasattr(args,'model_type'):
            self.config.update({"model_type": args.model_type})
        if hasattr(args,'aspect_ratio_path'):
            self.config.update({"aspect_ratio_path": args.aspect_ratio_path})
        if hasattr(args,'output_dir'):
            self.config.update({"output_dir": args.output_dir})
        


    def get_config(self, cfg, args=None):
        configs = yaml.load(open(str(cfg)), Loader=yaml.FullLoader)
        for k, v in configs.items():
            self.config.update(v)
        if args:
            self.update_config(args)
        return self.config


if __name__ == "__main__":
    config = Config()
