import torch
import torch.nn as nn
import src.parameters as pm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
labels = np.array([0, 0, 1])
print(labels)
positiveIndex = np.nonzero(labels)
negativeIndex = np.where(labels == 0)[0]
print(positiveIndex)
print(negativeIndex)

# %% tensors
label = np.int(1)
type(label)
labelt = torch.FloatTensor([label]).to(device)
labelt

# %% stuff to cuda
torch.cuda.empty_cache()
m =torch.nn.Linear(3, 3)
m.cuda()
next(m.parameters()).is_cuda
tensor = torch.FloatTensor([[[0,0,0]]], device=device)
tensor.is_cuda
# output will return "Error: expected object of type TYPEofINPUT
# but found type TYPEofNN", if they are not the same
output = m(tensor)
print(output)
z = torch.zeros(1, 1, 3, device=device)
z.is_cuda
z
torch.cat((output, z), dim=2)


# %% gru dim
params = pm.Params

encoder = nn.GRU(
    params.autoencoder.input_size,
    params.autoencoder.hidden_size,
    params.autoencoder.num_layers, batch_first=True).to(device)

input = torch.randn(5,1,params.autoencoder.input_size).to(device)
initialHidden = torch.randn(params.dim_y).to(device)
initialHidden = initialHidden.unsqueeze(0).unsqueeze(0)
initialHidden = torch.cat(
    (initialHidden, torch.zeros(1, 1, params.dim_z, device=device)), dim=2)
max_length = 5
hiddens = torch.zeros(
    max_length, 1, params.autoencoder.hidden_size, device=device)
print(hiddens.shape)
print(initialHidden.shape)
print(input.shape)
output, hn = encoder(input, initialHidden)

# %% criterions

criterion1 = nn.BCEWithLogitsLoss()
criterion2 = nn.BCEWithLogitsLoss().to(device)
criterion1.is_cuda
criterion2
# output will return "Error: expected object of type TYPEofNN
# but found type TYPEofINPUT", if they are not the same
output, hn = encoder(input, initialHidden)
print(output.shape)
print(hn.shape)
index = 3
hiddens[index, :, :] = hn
print(hiddens.shape)
hiddens = torch.cat((hn, hiddens), dim=0)
print(hiddens.shape)

z = torch.Tensor(device='cpu')
z.new_empty(1,1,1)
type(z)
print(hn[:,:,:5])
z = torch.cat((z, hn[:,:,:5]))

# %% squeeze
t = torch.FloatTensor([1, 2, 3])
t.shape
t.squeeze(0)
t.unsqueeze(1)
t.unsqueeze(0)
t.size(0)
t.squeeze(0)

# %% loss
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
input.grad
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
output.backward()
input.grad

torch.zeros(1, 1, 100, device='cpu')
help(torch.zeros)
