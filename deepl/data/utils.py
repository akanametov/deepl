from collections.abc import Sequence

class Dataset(Sequence):
    def __init__(self,):
        super().__init__()
        
    def __len__(self,):
        pass
    
    def __getitem__(self,idx):
        pass
    
class DataLoader(object):
    def __init__(self, dataset, batch_size):
        self.dataset=dataset
        self.batch_size=batch_size
        self.num_batch=len(dataset)//batch_size
        
    def __len__(self,):
        return self.num_batch
        
    def __iter__(self,):
        batch_size = self.batch_size
        for b in range(self.num_batch):
            idx = range(b*batch_size, (b+1)*batch_size)
            yield self.dataset[idx]