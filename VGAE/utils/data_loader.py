import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

def get_dataset(name, path='data'):
    """
    加载数据集并进行预处理（如：将图划分为训练、验证、测试边）
    """
    dataset = Planetoid(root=path, name=name, transform=T.NormalizeFeatures())
    data = dataset[0]

    # VGAE 主要是做链接预测，需要把边及逆行划分
    # transformer 会自动生成 data.tran_pos_edge_index, data.test_pos_edge_index等
    transform = T.RandomLinkSplit(
        num_val=0.05, 
        num_test=0.1,
        is_undirected=True,
        split_labels=True,
        )
    train_data, val_data, test_data = transform(data)
    return train_data, val_data, test_data
    
