import torch
import yaml
from torch_geometric.nn import VGAE
from model.vgae_nodel import GCNEncoder
from utils.data_loader import get_dataset

def main():
    # 1. 加载配置
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config['training']['divice'] if torch.cude.is_available() else "cpu")

    # 2. 准备数据
    train_data, val_data, test_data = get_dataset(config['data']['name'])
    train_data = train_data.to(device)

    # 3. 初始化模型
    encoder = GCNEncoder(
        config['model']['input_dim'],
        config['model']['hidden_dim'], 
        config['model']['latent_dim']
        )
    model = VGAE(encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])

    # 4. 训练循环
    for epoch in range(1, config['training']['epochs'] + 1):
        model.train()
        optimizer.zero_grad()

        z = model.encode(train_data.x, train_data.edge_index)

        loss = model.recon_loss(z, train_data.pos_edge_index)
        loss = loss + (1 / train_data.num_nodes) * model.kl_loss()

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                z = model.encode(test_data.x, test_data.edge_index)
                auc, ap = model.test(z, test_data.pos_edge_index, test_data.neg_edge_index)
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}, AUC: {auc:.4f}, AP: {ap:.4f}')

if __name__ == "__main__":
    main()