import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt


class XorDataset(Dataset):
    def __init__(self, size):
        X = []
        for i in range(int(np.sqrt(size))):
            for j in range(int(np.sqrt(size))):
                X.append([i,j])
        y = [x[0]^x[1] for x in X]
        self.X = torch.tensor(X, dtype= torch.float32)
        self.y = torch.tensor(y, dtype= torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx] , self.y[idx]

class XorRegressionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
                torch.nn.Linear(2, 2),
                torch.nn.ReLU(),
                torch.nn.Linear(2, 2),
                torch.nn.ReLU(),
                torch.nn.Linear(2, 2),
                torch.nn.ReLU(),
                torch.nn.Linear(2, 1),
                )

    def forward(self, x):
        return self.model(x)



model = XorRegressionModule()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
training_data = XorDataset(1_000)
training_loader = DataLoader(training_data)
val_data = XorDataset(1_000)
val_loader = DataLoader(val_data)
EPOCHS = 200
train_losses = []
val_losses = []
for epoch in range(EPOCHS):
    for i, data in enumerate(training_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outs = model(inputs)
        loss = loss_fn(outs, labels)
        loss.backward()
        optimizer.step()
    # testing
    losses = []
    with torch.no_grad():
        for i, data  in enumerate(val_loader):
            inputs, labels = data
            outs = model(inputs)
            val_loss = loss_fn(outs, labels)
            losses.append(val_loss.item())
        val_loss = np.mean(losses)
        print(f"in epoch {epoch} loss goes {loss.item()} and val loss {val_loss}")
        train_losses.append(loss.item())
        val_losses.append(np.mean(losses))

plt.plot(train_losses)
plt.plot(val_losses)
plt.show()
