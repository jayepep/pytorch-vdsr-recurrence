import torch.utils.data as data
import torch
import h5py
"""
定义数据集的读取方式，两个方法：getitem len
"""
class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get('data')
        self.target = hf.get('label')

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index, : , : , : ]).float(), torch.from_numpy(self.target[index,:,:,:]).float()
        
    def __len__(self):
        return self.data.shape[0]

if __name__ == '__main__':
    datahd = DatasetFromHdf5("data/train.h5")
    print(datahd[0])