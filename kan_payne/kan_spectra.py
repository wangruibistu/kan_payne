import sys
sys.path.append('/home/wangr/code/efficient-kan/src/')
from efficient_kan import KAN

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import utils

dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class myDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label


training_labels, training_spectra, validation_labels, validation_spectra = (
        utils.load_training_data()
    )
x_max = np.max(training_labels, axis = 0)
x_min = np.min(training_labels, axis = 0)

x = (training_labels - x_min)/(x_max - x_min) #- 0.5
x_valid = (validation_labels-x_min)/(x_max-x_min) #- 0.5

y_max = np.max(training_spectra, axis=0)
y_min = np.min(training_spectra, axis=0)

y = (training_spectra - y_min) / (y_max - y_min) #- 0.5
y_valid = (validation_spectra - y_min) / (y_max - y_min) #- 0.5


trainset = myDataset(
    torch.from_numpy(x).type(dtype=dtype).to(device=device),
    torch.from_numpy(y).type(dtype=dtype).to(device=device),
)
valset = myDataset(
    torch.from_numpy(x_valid).type(dtype=dtype).to(device=device),
    torch.from_numpy(y_valid).type(dtype=dtype).to(device=device),
)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)

dim_in = training_labels.shape[1]
dim_out = training_spectra.shape[1]

model = KAN(
    layers_hidden=[dim_in, 64, 256, dim_out],
    base_activation=torch.nn.GELU
    )
model.to(device)

optimizer = optim.AdamW(
    model.parameters(), 
    lr=1e-4, 
    weight_decay=1e-4,
    )
# scheduler = optim.lr_scheduler.ExponentialLR(
#     optimizer, gamma=0.8
#     )

# Define loss
# criterion = nn.CrossEntropyLoss()
loss_fn = torch.nn.L1Loss(reduction = 'mean')
training_loss =[]
validation_loss = []

for epoch in range(10000):
    # Train
    model.train()
    train_loss = 0
    with tqdm(trainloader) as pbar:
        for i, (labels, spectra) in enumerate(pbar):
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(labels)
            loss = loss_fn(output, spectra.to(device)) 
            train_loss += loss.data.item() * 1e4
            loss.backward()
            optimizer.step()
            # accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
            # if i % 100 == 0:
        # pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

    train_loss /= len(trainloader)
    # Validation
    model.eval()
    val_loss = 0
    # val_accuracy = 0
    with torch.no_grad():
        for labels, spectra in valloader:
            labels = labels.to(device)
            output = model(labels)
            val_loss += loss_fn(output, spectra.to(device)).data.item() *1e4
            # val_accuracy += (
            #    (output.argmax(dim=1) == labels.to(device)).float().mean().item()
            # )
    val_loss /= len(valloader)
    # val_accuracy /= len(valloader)
    training_loss.append(train_loss)
    validation_loss.append(val_loss)
    # Update learning rate
    # scheduler.step()
    # if epoch % 100 ==  0:
    print(
        f"Epoch {epoch + 1}, Train Loss: {train_loss}, Val Loss: {val_loss}"
    )
    
np.savez(
    "training_loss.npz", 
    training_loss=training_loss, 
    validation_loss=validation_loss
)
torch.save(model, './model_save/Payne_KAN_model_01.kpt')

