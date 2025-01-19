import pytorch_lightning as pl
from pytorch_lightning import callbacks
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint



class XorRegression(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
                torch.nn.Linear(2,4),
                torch.nn.Sigmoid(),
                torch.nn.Linear(4,1)
                )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        out = self(inputs)
        loss = self.loss(out, labels)
        self.log("loss", loss, prog_bar = True, on_step = False, on_epoch=True)
        return loss


class XorDataset(Dataset):
    def __init__(self, lenght):
        X = []
        for i in range(int(np.sqrt(lenght))):
            for j in range(int(np.sqrt(lenght))):
                X.append([i,j])
        y = [ [x[0]^x[1]] for x in X]
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


dataset = XorDataset(10_000)
dataloader = DataLoader(dataset, num_workers = 3)
model = XorRegression()
checkpoint = ModelCheckpoint()
trainer = pl.Trainer(max_epochs = 200,  callbacks = [checkpoint])
trainer.fit(model, dataloader)
print(checkpoint.best_model_path)
