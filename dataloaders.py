import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


class MyCollate:
    
    '''
    class to add padding to the batches
    collat_fn in dataloader is used for post processing on a single batch. Like __getitem__ in dataset class
    is used on single example
    '''
    
    def __init__(self, pad_idx, return_protected_attribute):
        self.pad_idx = pad_idx
        self.return_protected_attribute = return_protected_attribute
        
    #__call__: a default method
    ##   First the obj is created using MyCollate(pad_idx) in data loader
    ##   Then if obj(batch) is called -> __call__ runs by default
    def __call__(self, batch):
        
        source = [item[0] for item in batch] 
        source = pad_sequence(source, batch_first=False, padding_value = self.pad_idx) 
        
        target = [item[1] for item in batch]
        
        if self.return_protected_attribute:
            prot_attr = [item[2] for item in batch]
            return source, torch.tensor(target), torch.tensor(prot_attr)
        else:
            return source, torch.tensor(target)
        


def get_train_loader(dataset, batch_size, return_protected_attribute=False, num_workers=0, shuffle=False, pin_memory=True):
    # get pad_idx for collate fn
    pad_idx = dataset.vocab.stoi['<PAD>']
    
    # define loader
    loader = DataLoader(
        dataset, 
        batch_size = batch_size, 
        shuffle = shuffle,
        collate_fn = MyCollate(pad_idx = pad_idx, return_protected_attribute=return_protected_attribute)  # MyCollate class runs __call__ method by default
    )
    return loader


def get_inference_loader(dataset, train_dataset, batch_size, return_protected_attribute=False, num_workers=0, shuffle=False, pin_memory=True):
    pad_idx = train_dataset.vocab.stoi['<PAD>']
    loader = DataLoader(
        dataset, 
        batch_size = batch_size, 
        shuffle = shuffle,
        collate_fn = MyCollate(pad_idx = pad_idx, return_protected_attribute=return_protected_attribute))
    return loader