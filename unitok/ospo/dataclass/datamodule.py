import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from unitok.ospo.utils.common import read_json
from unitok.ospo.dataclass.datasets_v2 import PreferenceDataset

def collate_identity(batch):
    return batch

class TrainDataModule(pl.LightningDataModule):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        self.train_dataset = None
        self.val_datasets = None

    def setup(self):
        self.train_dataset = PreferenceDataset(
                                        seed=self.config.experiment.seed,
                                        data_path=self.config.dataset.train.data_path,
                                        tokenizer=self.tokenizer,
                                        num_samples=self.config.dataset.train.num_samples,
                                        copo_mode=True if self.config.algo.copo_weight > 0 else False,
                                        use_mask=self.config.use_mask,
                                        mask_dir=self.config.mask_dir,
                                        )


    def train_dataloader(self):
        if self.train_dataset is not None:
            return DataLoader(
                    dataset=self.train_dataset, 
                    shuffle=True,
                    collate_fn=self.train_dataset.collate_fn,            # lambda batch: batch, 
                    batch_size=self.config.dataset.train.batch_size,     # Batch size per process
                    num_workers=self.config.dataset.train.num_workers,   # Number of data loading workers
                    )
    
    def val_dataloader(self):
        return None
        # if self.val_datasets is not None:
        #     return [DataLoader(
        #             dataset=val_dataset, 
        #             shuffle=False,
        #             # collate_fn=lambda batch: batch,                      # Use different collate_fn
        #             collate_fn=val_dataset.collate_fn,            
        #             batch_size=self.config.dataset.val.batch_size,         # Batch size per process
        #             num_workers=self.config.dataset.val.num_workers,       # Number of data loading workers
        #             drop_last=True
        #             ) for val_dataset in self.val_datasets]
