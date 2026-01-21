import pandas as pd
import numpy as np
import torch
import networkx as nx
import os
from torch_geometric.data import Data

OUTPUT_DIR = 'data'

def gerar_e_salvar_tabular(n_samples=5000):
    print("Gerando dados Tabulares...")
    np.random.seed(42)
    y = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    val = np.random.exponential(scale=100, size=n_samples)
    mouse_speed = np.random.normal(0.5, 0.1, size=n_samples)
    val[y==1] = np.random.exponential(scale=110, size=np.sum(y==1))
    mouse_speed[y==1] = np.random.normal(0.9, 0.05, size=np.sum(y==1))
    
    df = pd.DataFrame({'valor': val, 'mouse_speed': mouse_speed, 'label': y})
    file_path = os.path.join(OUTPUT_DIR, 'tabular_data.csv')
    df.to_csv(file_path, index=False)
    print(f"Salvo em: {file_path}")

def gerar_e_salvar_sequencia(n_samples=500, seq_len=30):
    print("Gerando dados Sequenciais (LSTM)...")
    np.random.seed(42)
    data = []
    for _ in range(n_samples):
        t = np.linspace(0, 3, seq_len)
        seq = np.stack([t, np.sin(t) + np.random.normal(0, 0.1, seq_len)], axis=1)
        data.append(seq)
    
    tensor_data = torch.FloatTensor(np.array(data, dtype=np.float32))
    file_path = os.path.join(OUTPUT_DIR, 'lstm_data.pt')
    torch.save(tensor_data, file_path)
    print(f"Salvo em: {file_path}")

def gerar_e_salvar_grafo(num_nodes=200):
    print("Gerando dados de Grafo (GNN)...")
    np.random.seed(42)
    G = nx.watts_strogatz_graph(n=num_nodes, k=4, p=0.1)
    x = torch.rand((num_nodes, 2))
    y = torch.zeros(num_nodes, dtype=torch.long)
    
    for i in range(10):
        y[i] = 1
        x[i] = torch.tensor([0.9, 0.9])
        if i > 0:
            G.add_edge(0, i)

    edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    data_obj = Data(x=x, edge_index=edge_index, y=y)
    
    file_path = os.path.join(OUTPUT_DIR, 'gnn.pt') 
    torch.save(data_obj, file_path)
    print(f"Salvo em: {file_path}")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    gerar_e_salvar_tabular()
    gerar_e_salvar_sequencia()
    gerar_e_salvar_grafo()
    print("--- Geração de Dados Concluída ---")
