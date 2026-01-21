import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# --- 1. Autoencoder (Tabular) ---
class TabularAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(TabularAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 4), # Latent
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.Tanh(),
            nn.Linear(8, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# --- 2. LSTM Autoencoder (Sequencial) ---
class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dim=64, latent_dim=16):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.encoder_lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_dim, batch_first=True)
        self.hidden2latent = nn.Linear(hidden_dim, latent_dim)
        self.latent2hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        latent = self.hidden2latent(h_n[-1])
        hidden = self.latent2hidden(latent)
        decoder_input = hidden.unsqueeze(1).repeat(1, self.seq_len, 1)
        decoder_out, _ = self.decoder_lstm(decoder_input)
        reconstruction = self.output_layer(decoder_out)
        return reconstruction

# --- 3. GCN ---
class FraudGCN(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 2) # Saída binária (Fraude ou Não)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
