import matplotlib.pyplot as plt
import torch
import warnings
from data_processing import create_pytorch_geometric_graph_data_list_from_smiles_and_labels
from model import GCN
import os
from scipy.signal import find_peaks
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Define the SMILES string
smiles = ''

# Create PyTorch Geometric graph data list from SMILES and labels
data = create_pytorch_geometric_graph_data_list_from_smiles_and_labels([smiles], [1])

# Load the pre-trained model
model = GCN()
model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
model.eval()

# Perform inference
with torch.no_grad():
    output = list(model(data[0].x, data[0].edge_index, None, data[0].edge_attr)[0][0])

# Invert the y-values
inverted_y_values = [1.0 - y for y in output]

# Plot the predicted IR spectrum
plt.plot(range(0, 1000), inverted_y_values, color='red')
plt.title('Predicted IR Spectrum of ' + smiles)
plt.xlabel('Wavenumber (cm-1)')
plt.ylabel('Transmittance')
plt.xlim(0, 1000)
plt.xticks([0, 200, 400, 600, 800, 1000], ['4000', '3000', '2000', '1000', '500', '400'])

# Find the peaks in the spectrum
peaks, _ = find_peaks(output, height=0.24)
peak = [round(4000 - (3.6 * line)) for line in peaks]
# Show the plot
plt.show()

