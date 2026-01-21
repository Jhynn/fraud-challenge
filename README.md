# 🛡️ Sistema Multi-Modal de Detecção de Fraude (Behavioral & Graph)

Este projeto implementa um pipeline completo de Engenharia de Machine Learning para detecção de fraudes financeiras. Diferente de abordagens tradicionais focadas apenas em regras ou dados tabulares, este sistema utiliza uma arquitetura **multi-modal** que analisa:
1.  **Dados Transacionais:** Valores e horários.
2.  **Biometria Comportamental:** Dinâmica de uso do mouse (velocidade, trajetória).
3.  **Redes de Relacionamento:** Grafos para detectar anéis de fraude (Money Mules).



## 📂 Estrutura do Projeto

O projeto segue princípios de **Modularidade** e **Reprodutibilidade**, separando a geração de dados, treinamento e inferência.

```text
fraud-challenge/
│
├── data/                   # Armazenamento de datasets (CSV, .pt) - Gerados pelo script
├── models/                 # Artefatos serializados (.pkl, .pth) prontos para produção
├── notebooks/              # Análises Exploratórias (EDA) e Storytelling
│   └── eda.ipynb
│
├── src/                    # Código Fonte Modular
│   ├── __init__.py
│   ├── data_gen.py         # Pipeline de Geração de Dados Sintéticos (ETL)
│   ├── models_def.py       # Definições de Arquiteturas PyTorch (LSTM, GNN, Autoencoder)
│   ├── train.py            # Script de treinamento CLI e persistência
│   └── inference.py        # Motor de inferência (Lazy Loading & Previsão)
│
├── api.py                  # API REST (FastAPI) para servir os modelos
├── requirements.txt        # Dependências do projeto
├── Dockerfile              # Configuração para Containerização
└── README.md               # Documentação

```

## 🧠 Modelos Implementados

O sistema oferece 5 estratégias de defesa, variando de modelos explicáveis a Deep Learning avançado:

| Modelo | Tipo | Aplicação Principal |
| --- | --- | --- |
| **Random Forest** | Supervisionado (Tabular) | Baseline robusto. Analisa valor e velocidade média do mouse. |
| **Isolation Forest** | Não-Supervisionado | Detecção de anomalias (Outliers) sem necessidade de labels de fraude prévios. |
| **Autoencoder** | Deep Learning (Tabular) | Reconstrói padrões normais. Alto erro de reconstrução indica fraude complexa. |
| **LSTM** | Time Series (Sequencial) | Analisa a **trajetória** do mouse (30 pontos). Detecta movimentos robóticos/lineares. |
| **GCN (Graph Convolutional Neural Network)** | Grafos (Relacional) | Analisa a topologia da rede. Detecta nós conectados a anéis de fraude. |

## 🚀 Instalação e Configuração

### Pré-requisitos

* Python 3.8+
* PIP

### 1. Clonar e Instalar Dependências

```bash
git clone https://github.com/jhynn/fraud-challenge.git
cd fraud-challenge
pip install -r requirements.txt

```

---

## ⚙️ Pipeline de Execução

Siga a ordem lógica do pipeline de dados para reproduzir o ambiente.

### Passo 1: Geração de Dados (ETL)

O script abaixo cria dados sintéticos ricos (simulando padrões de horário e comportamento robótico) e salva na pasta `data/`.

```bash
python -m src.data_gen

```

### Passo 2: Treinamento dos Modelos

Você pode treinar modelos individualmente. Os artefatos serão salvos em `models/`.

```bash
# Treinar Baseline e Anomalias
python -m src.train --model rf              # random forest
python -m src.train --model iso_forest      # isolation forest

# Treinar Deep Learning
python -m src.train --model autoencoder
python -m src.train --model lstm
python -m src.train --model gcn

```

### Passo 3: Subir a API (Produção)

O projeto expõe os modelos através de uma API REST de alta performance usando **FastAPI** e servidor **Uvicorn**.

```bash
uvicorn api:app --reload

```

Acesse a documentação interativa (Swagger UI) em:
👉 **http://127.0.0.1:8000/docs**


JSON de exemplos.

Para `POST /predict/lstm`

```text
{
  "trajetoria": [
    [0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4],
    [0.5, 0.5], [0.6, 0.6], [0.7, 0.7], [0.8, 0.8], [0.9, 0.9],
    [1.0, 1.0], [1.1, 1.1], [1.2, 1.2], [1.3, 1.3], [1.4, 1.4],
    [1.5, 1.5], [1.6, 1.6], [1.7, 1.7], [1.8, 1.8], [1.9, 1.9],
    [2.0, 2.0], [2.1, 2.1], [2.2, 2.2], [2.3, 2.3], [2.4, 2.4],
    [2.5, 2.5], [2.6, 2.6], [2.7, 2.7], [2.8, 2.8], [2.9, 2.9]
  ]
}
```

Para `POST /predict/gcn`

```text
{
  "features": [0.5, 0.5],
  "vizinhos_ids": [0, 1, 2]
}
```

---

## 🐳 Executando com Docker

Para garantir isolamento total do ambiente, o projeto pode ser executado via Docker.

```bash
# 1. Construir a imagem (Isso rodará o treino automaticamente durante o build)
docker build -t fraud-detection-system .

# 2. Rodar o container
docker run -p 8000:8000 fraud-detection-system

```

## 📊 Análise Exploratória

Para entender as hipóteses por trás dos modelos (ex: por que fraudes ocorrem de madrugada?), consulte o notebook:
`notebooks/eda.ipynb`

## 👨‍💻 Autor

Desenvolvido a fim de simular um projeto real de _data science_, demonstrando proficiência em:

* **Python Engineering** (Modularização, OOP)
* **MLOps** (Pipeline, API, Docker)
* **Deep Learning** (PyTorch, PyG)
