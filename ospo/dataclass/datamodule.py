import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.utils.common import read_json
from ospo.dataclass import PreferenceDataset, ValidationDataset, BaseDataset, NegativeDataset 

# janus_fa 환경 실행 시 lambda 의 대체재
def identity_collate(batch):
    return batch

class TrainDataModule(pl.LightningDataModule):
    def __init__(self, config, chat_processor, image_processor, tokenizer, train_shuffle=True, val_shuffle=False):
        self.config = config
        self.tokenizer = tokenizer
        self.chat_processor = chat_processor
        self.image_processor = image_processor
        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle

        self.train_dataset = None
        self.val_datasets = None
        self.copo_mode = True if config.algo.copo_weight > 0.0 else False
        print(f"Loading Train datamodule in CoPO mode: {self.copo_mode}")

    def setup(self):
        self.train_dataset = PreferenceDataset(
                                        seed=self.config.experiment.seed,
                                        data_path=self.config.dataset.train.data_path,
                                        chat_processor=self.chat_processor,
                                        image_processor=self.image_processor,
                                        tokenizer=self.tokenizer,
                                        num_samples=self.config.dataset.train.num_samples,
                                        copo_mode=self.copo_mode,
                                        use_mask=self.config.use_mask,
                                        mask_dir=self.config.mask_dir,
                                        off_policy=self.config.off_policy
                                        )
        if self.config.dataset.get('val', None) is not None:
            val_data_path_dict = self.config.dataset.val.data_path
            for k, v in val_data_path_dict.items():
                if v is None:
                    continue
                task_id = int(k.split("_")[1])
                if self.val_datasets is None:
                    self.val_datasets = []

                self.val_datasets.append(
                    ValidationDataset(data_path=v,
                                    chat_processor=self.chat_processor,
                                    image_processor=self.image_processor,
                                    tokenizer=self.tokenizer,
                                    task_id=task_id,
                                    copo_mode=self.copo_mode)
                )


    def train_dataloader(self):
        if self.train_dataset is not None:
            return DataLoader(
                    dataset=self.train_dataset, 
                    shuffle=self.train_shuffle,  
                    collate_fn=self.train_dataset.collate_fn,            # lambda batch: batch, 
                    batch_size=self.config.dataset.train.batch_size,     # Batch size per process
                    num_workers=self.config.dataset.train.num_workers,   # Number of data loading workers
                    )
    
    def val_dataloader(self):
        if self.val_datasets is not None:
            # 데이터로더의 리스트
            return [DataLoader(
                    dataset=val_dataset, 
                    shuffle=self.val_shuffle,
                    # collate_fn=lambda batch: batch,                     # Use different collate_fn
                    collate_fn=val_dataset.collate_fn,            
                    batch_size=self.config.dataset.val.batch_size,       # Batch size per process
                    num_workers=self.config.dataset.val.num_workers,     # Number of data loading workers
                    drop_last=True
                    ) for val_dataset in self.val_datasets]
        else:
            return None


class GenerationDataModule(pl.LightningDataModule):
    def __init__(self, config, step: float = 1.0): # step in framework
        if step not in [1.0, 2.0, 3.1, 3.2, 4.0]: 
            raise ValueError("Step must be one of [1, 2, 3, 4].")
        self.config = config
    
        # Element/Base prompt generation
        if step == 1.0:
            if config.max_len is None:
                if config.category == "object":
                    max_len = 120
                elif config.category == "2D_spatial" or config.category == "3D_spatial":  # "spatial":
                    max_len = 3000 # 40
                elif config.category == "non-spatial" or config.category == "complex":
                    max_len = 4000
                elif config.category == "numeracy1" or config.category == "numeracy2":
                    max_len = 3000
                else:
                    max_len = 2000 # 70
            else:
                max_len = config.max_len
            self.dataset = list(range(max_len)) # dummy data

        elif step == 3.2:
            self.dataset = NegativeDataset(base_path=config.data_path.merge_standard, negative_qs_path=config.data_path.merge_nqs) 
            # Do not use s_idx and e_idx

        else:
            self.dataset = BaseDataset(fpath=config.data_path,
                                       s_idx=config.s_idx,
                                       e_idx=config.e_idx)


    def gen_dataloader(self):
        return DataLoader(self.dataset, 
                        batch_size=self.config.batch_size, 
                        collate_fn=identity_collate,
                        num_workers=self.config.num_workers,
                        pin_memory=True,
                        drop_last=False,
                        shuffle=False)
        
class NewGenerationDataModule(pl.LightningDataModule):
    def __init__(self, config, step:int=1): # step in framework
        if step not in [1, 2, 3, 4]: 
            raise ValueError("Step must be one of [1, 2, 3, 4].")
        self.config = config
    
        # Element/Base prompt generation
        if step == 1:
            '''
            if config.max_len is None:
                if config.category == "object":
                    max_len = 120
                elif config.category == "spatial":
                    max_len = 40
                elif config.category == "non-spatial" or config.category == "complex":
                    max_len = 4000
                else:
                    max_len = 70
            else:
                max_len = config.max_len
            '''
            if config.max_len is None:
                max_len = 6000
            else:
                max_len = config.max_len
            
            self.dataset = list(range(max_len)) # dummy data


        # Dense/Negative prompt generation and Image generation
        elif step == 2 or step == 3 or step == 4:
            self.dataset = NewBaseDataset(fpath=config.data_path)
        
        # Question/Answer generation
        elif step == 5:
            self.dataset = read_json(config.data_path) 


    def gen_dataloader(self):
        return DataLoader(self.dataset, # debug point
                        batch_size=self.config.batch_size, 
                        collate_fn=identity_collate, # identity_collate
                        num_workers=self.config.num_workers,
                        pin_memory=True,
                        drop_last=False,
                        shuffle=False)
                        

class VerifyDataModule(pl.LightningDataModule):
    def __init__(self, config, chat_processor, image_processor, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.chat_processor = chat_processor
        self.image_processor = image_processor

        self.dataset = None

    # TODO: verify in config.yaml
    def setup(self):
        self.dataset = PreferenceWithPerturbationDataset(
                                        seed=self.config.experiment.seed,
                                        data_path=self.config.dataset.verify.data_path,
                                        chat_processor=self.chat_processor,
                                        image_processor=self.image_processor,
                                        tokenizer=self.tokenizer,
                                        num_samples=self.config.dataset.verify.num_samples,
                                        # omit sampling_rate
                                        )

    def dataloader(self):
        if self.dataset is not None:
            return DataLoader(
                    dataset=self.dataset, 
                    shuffle=False,
                    collate_fn=self.dataset.collate_fn,            # lambda batch: batch, 
                    batch_size=self.config.dataset.verify.batch_size,     # Batch size per process
                    num_workers=self.config.dataset.verify.num_workers,   # Number of data loading workers
                    )

