import argparse
import joblib
import os
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models_def import TabularAutoencoder, LSTMAutoencoder, FraudGCN

DATA_DIR = 'data'
MODELS_DIR = 'models'

def load_tabular():
    path = os.path.join(DATA_DIR, 'tabular_data.csv')
    return pd.read_csv(path)

def load_sequence():
    path = os.path.join(DATA_DIR, 'lstm_data.pt')
    return torch.load(path, weights_only=False)

def load_graph():
    path = os.path.join(DATA_DIR, 'gnn.pt') 
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo {path} não encontrado. Rode src.data_gen primeiro.")
    return torch.load(path, weights_only=False)

def train(model_name):
    print(f"--- Iniciando Treinamento: {model_name.upper()} ---")
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    if model_name in ['rf', 'iso_forest']:
        df = load_tabular()
        X = df[['valor', 'mouse_speed']]
        y = df['label']
        
        if model_name == 'rf':
            model = RandomForestClassifier(class_weight='balanced', random_state=42)
            pipeline = Pipeline([
                ('scaler', StandardScaler()), 
                ('model', model)
            ])
            pipeline.fit(X, y)

        elif model_name == 'iso_forest':
            model = IsolationForest(contamination=0.1, random_state=42)
            pipeline = Pipeline([
                ('scaler', StandardScaler()), 
                ('model', model)
            ])
            pipeline.fit(X)

        joblib.dump(pipeline, f'{MODELS_DIR}/{model_name}.pkl') # type: ignore
        print(f"Salvo em {MODELS_DIR}/{model_name}.pkl")

    elif model_name == 'autoencoder':
        df = load_tabular()
        X = df[df['label'] == 0][['valor', 'mouse_speed']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled)

        net = TabularAutoencoder(input_dim=2)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        for epoch in range(50):
            optimizer.zero_grad()
            output = net(X_tensor)
            loss = criterion(output, X_tensor)
            loss.backward()
            optimizer.step()

        state = {'state_dict': net.state_dict(), 'scaler': scaler}
        torch.save(state, f'{MODELS_DIR}/autoencoder.pth')
        print(f"Salvo em {MODELS_DIR}/autoencoder.pth")

    elif model_name == 'lstm':
        tensor_data = load_sequence()
        net = LSTMAutoencoder(seq_len=30, n_features=2)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        for epoch in range(20):
            optimizer.zero_grad()
            output = net(tensor_data)
            loss = criterion(output, tensor_data)
            loss.backward()
            optimizer.step()
            
        torch.save(net.state_dict(), f'{MODELS_DIR}/lstm.pth')
        print(f"Salvo em {MODELS_DIR}/lstm.pth")

    elif model_name == 'gcn':
        data = load_graph()
        net = FraudGCN(num_features=2)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        
        net.train()
        for epoch in range(100):
            optimizer.zero_grad()
            out = net(data)
            loss = torch.nn.functional.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            
        state = {'state_dict': net.state_dict(), 'base_graph': data}
        torch.save(state, f'{MODELS_DIR}/gcn.pth')
        print(f"Salvo em {MODELS_DIR}/gcn.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', 
        type=str, 
        required=True, 
        choices=['rf', 'iso_forest', 'autoencoder', 'lstm', 'gcn']
    )
    args = parser.parse_args()
    train(args.model)
