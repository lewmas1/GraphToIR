import torch
import matplotlib.pyplot as plt
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import pickle
from model import GCN

with open('G2I.pkl', 'rb') as f:
    data = pickle.load(f)

model = GCN()
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

loss_fn = torch.nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00025)

# Use GPU for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Wrap data in a data loader
data_size = len(data)
NUM_GRAPHS_PER_BATCH = 75
loader = DataLoader(data[:int(data_size* 0.8)], batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
test_loader = DataLoader(data[int(data_size * 0.8):], batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)


def train(data):
    """
    Trains the model using the given data.

    Args:
        data: The input data for training.

    Returns:
        loss: The loss value after training.
        embedding: The embedding generated by the model.
    """
    # Enumerate over the data
    for batch in loader:

      batch.to(device)
      # Reset gradients
      optimizer.zero_grad()
      # Passing the node features and the connection info
      pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch, batch.edge_attr)
      # Calculating the loss and gradients
      loss = loss_fn(pred, (batch.y))
      loss.backward()
      # Update using the gradients
      optimizer.step()
    return loss, embedding

def test(data):
    for batch in test_loader:
        batch.to(device)
        pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch, batch.edge_attr)
        loss = loss_fn(pred, (batch.y))
        return loss

print("Starting training...")
losses = []
testlosses=[]

for epoch in range(150):
    model.train()
    loss, h = train(data)
    testloss = test(data)
    losses.append(loss)
    testlosses.append(testloss)
    print(f'Epoch: {epoch}, Train loss: {loss.item()}, test loss: {testloss.item()}')
    torch.save(model.state_dict(), ('model.pt'))

# Display the loss
# plt.plot([i for i in range(NUM_EPOCHS * BATCH_NUMBER)], train_losses, label='Training loss')
plt.plot(range(150), (losses), label='Training loss')
plt.show()