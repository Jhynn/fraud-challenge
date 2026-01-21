import joblib
import torch
import numpy as np
import pandas as pd
from src.models_def import TabularAutoencoder, LSTMAutoencoder, FraudGCN

class MultiModelPredictor:
    def __init__(self):
        self.models = {}
    
    def load_model(self, model_type):
        if model_type in self.models:
            return
        
        path = f'models/{model_type}'
        
        if model_type == 'rf':
            self.models['rf'] = joblib.load(path + '.pkl')
        elif model_type == 'iso_forest':
            self.models['iso_forest'] = joblib.load(path + '.pkl')
        elif model_type == 'autoencoder':
            checkpoint = torch.load(path + '.pth')
            net = TabularAutoencoder(input_dim=2)
            net.load_state_dict(checkpoint['state_dict'])
            net.eval()
            self.models['autoencoder'] = {'net': net, 'scaler': checkpoint['scaler']}
        elif model_type == 'lstm':
            net = LSTMAutoencoder(seq_len=30, n_features=2)
            net.load_state_dict(torch.load(path + '.pth'))
            net.eval()
            self.models['lstm'] = net

        elif model_type == 'gcn':
            checkpoint = torch.load(path + '.pth', weights_only=False)
            net = FraudGCN(num_features=2)
            net.load_state_dict(checkpoint['state_dict'])
            net.eval()
            self.models['gcn'] = {'net': net, 'base_data': checkpoint['base_graph']}

    def predict(self, model_type, input_data):
        self.load_model(model_type)
        
        if model_type == 'rf':
            df = pd.DataFrame([input_data], columns=['valor', 'mouse_speed'])
            prob = self.models['rf'].predict_proba(df)[0][1]
            return {"fraude_prob": float(prob)}

        elif model_type == 'iso_forest':
            df = pd.DataFrame([input_data], columns=['valor', 'mouse_speed'])
            pred = self.models['iso_forest'].predict(df)[0]
            return {"status": "ANOMALIA" if pred == -1 else "NORMAL", "score": float(self.models['iso_forest'].decision_function(df)[0])}

        elif model_type == 'autoencoder':
            net = self.models['autoencoder']['net']
            scaler = self.models['autoencoder']['scaler']
            x_scaled = scaler.transform([input_data])
            x_tensor = torch.FloatTensor(x_scaled)
            with torch.no_grad():
                reconstruction = net(x_tensor)
                mse = torch.mean((x_tensor - reconstruction)**2).item()
            return {"erro_reconstrucao": mse, "fraude": mse > 1.0}
            
        elif model_type == 'lstm':
            net = self.models['lstm']
            t_data = torch.FloatTensor([input_data])
            with torch.no_grad():
                reconstruction = net(t_data)
                mse = torch.mean((t_data - reconstruction)**2).item()
            return {"erro_sequencia": mse, "comportamento_robo": mse > 0.1}

        # --- MUDANÇA AQUI: gcn ---
        elif model_type == 'gcn':
            return {"status": "GCN Processada (Mock)", "risco_rede": "ALTO"}
