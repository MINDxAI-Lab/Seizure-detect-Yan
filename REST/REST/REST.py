import torch
import torch.nn.functional as F
from torch.nn import Linear
import torch.optim as optim
from torch_geometric.nn import GraphConv, global_mean_pool
import lightning as L
from sklearn.metrics import f1_score

class LitREST(L.LightningModule):
    def __init__(self, fire_rate, multi, T):
        super().__init__()
        self.fire_rate = fire_rate
        self.multi = multi
        self.T = T

        # input dim = 100 (T=1, FFT→100 bins), hidden = 32
        self.l1 = Linear(100, 32, bias=False)
        self.l2 = Linear(32, 32)

        # two graph-conv layers
        self.gc1 = GraphConv(32, 32)
        self.gc2 = GraphConv(32, 32)

        # final readout
        self.fc  = torch.nn.Linear(32, 1)

        # enable high-precision matmul on Ampere+
        torch.set_float32_matmul_precision('high')

    def update(self, x_t, edge_index, edge_weight, s_t, fire_rate):
        # feedforward + recurrent
        if s_t is None:
            s_t = self.l1(x_t)
        else:
            s_t = self.l1(x_t) + self.l2(s_t)

        # two graph convolutions with spiking mask
        ds = self.gc1(s_t, edge_index, edge_weight.float())
        ds = ds.relu()
        ds = self.gc2(ds, edge_index, edge_weight.float())

        mask = torch.rand(ds.size(), device=ds.device) <= fire_rate
        ds = ds * mask

        return s_t + ds

    def forward(self, data):
        # data.x shape [num_nodes, T, freq_bins]
        # z-score normalize along time/freq
        clip = (data.x.float() - data.x.float().mean(2, keepdim=True)) / (
            data.x.float().std(2, keepdim=True) + 1e-10
        )

        s_t = None
        for t in range(self.T):
            N = torch.randint(1, 10, (1,)).item() if self.multi else 1
            for _ in range(N):
                x_t = clip[:, t, :].float()
                s_t = self.update(
                    x_t,
                    data.edge_index,      # from adj_mx_3d.pkl
                    data.edge_weight,     # from adj_mx_3d.pkl
                    s_t,
                    self.fire_rate
                )
        return self.fc(s_t)

    def training_step(self, data, batch_idx):
        # standard Lightning training step
        s   = self(data)
        out = global_mean_pool(s, data.batch)
        y_prob = torch.sigmoid(out.reshape(-1, 1))
        y_true = data.y.type(torch.float32).reshape(-1, 1)
        
        loss = F.mse_loss(y_prob, y_true)
        self.log('train_loss', loss, prog_bar=True)
        
        # Calculate F1 score
        y_pred = (y_prob >= 0.5).float()
        y_true_cpu = y_true.detach().cpu().numpy()
        y_pred_cpu = y_pred.detach().cpu().numpy()
        
        if len(set(y_true_cpu.flatten())) > 1:  # Check if both classes are present
            train_f1 = f1_score(y_true_cpu, y_pred_cpu, average='binary', zero_division=0)
            self.log('train_f1', train_f1, prog_bar=True)
        
        return loss

    def validation_step(self, data, batch_idx):
        # validation step to track validation loss and F1
        s   = self(data)
        out = global_mean_pool(s, data.batch)
        y_prob = torch.sigmoid(out.reshape(-1, 1))
        y_true = data.y.type(torch.float32).reshape(-1, 1)
        
        loss = F.mse_loss(y_prob, y_true)
        self.log('val_loss', loss, prog_bar=True)
        
        # Calculate F1 score
        y_pred = (y_prob >= 0.5).float()
        y_true_cpu = y_true.detach().cpu().numpy()
        y_pred_cpu = y_pred.detach().cpu().numpy()
        
        if len(set(y_true_cpu.flatten())) > 1:  # Check if both classes are present
            val_f1 = f1_score(y_true_cpu, y_pred_cpu, average='binary', zero_division=0)
            self.log('val_f1', val_f1, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [1000, 2000], gamma=0.3
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
