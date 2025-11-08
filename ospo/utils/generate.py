
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelSummary
from eval.eval_dataset import *
from torch.utils.data import DataLoader

DATASET_MAPPING_DICT = {
    "t2icompbench": T2ICompBench,
    "dpgbench": DPGEval,
    "geneval": GenEval
}

def get_trainer(device, world_size, precision="bf16"):    
    trainer = Trainer(
        accelerator=device,
        devices=world_size,
        strategy="ddp",
        max_epochs=1, 
        precision=precision,
        callbacks=[ModelSummary(max_depth=2)],
    )

    return trainer


def get_dataloader(config, args):
    dataset_cls = DATASET_MAPPING_DICT.get(config.base.task_name, None)
    if dataset_cls is None:
        raise ValueError(f"config.base.task_name must be one of ['t2icompbench', 'dpgbench', 'geneval']. But, you gave {config.base.task_name}")
    
    category_list = args.category
    dataset = dataset_cls(data_path=config.task.data_path,
                        category_list=config.task.category if category_list is None else category_list,
                        split=config.task.split,
                        s_idx=args.s_idx,
                        e_idx=args.e_idx)
    
    dataloader = DataLoader(
        dataset,
        collate_fn=lambda batch: batch,
        batch_size=config.task.batch_size,
        num_workers=config.task.num_workers,
        pin_memory=True,
        drop_last=False
        ) 
    return dataloader