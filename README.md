# SmartLeito

**Gestão inteligente de leitos hospitalares**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-F7931E?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/status-prot%C3%B3tipo%20acad%C3%AAmico-lightgrey)

O SmartLeito é um projeto acadêmico (Unimar) que usa Machine Learning para **prever a data de alta de pacientes** e a **demanda futura de leitos**, apresentando tudo em um **dashboard de gestão**.

> **Aviso:** este é um protótipo educacional que utiliza dados sintéticos. Não deve ser usado para decisões clínicas reais.

---

## Sumário

- [Funcionalidades](#funcionalidades)
- [Stack](#stack)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Como rodar](#como-rodar)
- [Endpoints da API](#endpoints-da-api)
- [Solução de problemas](#solução-de-problemas)
- [Equipe](#equipe)

---

## Funcionalidades

- Cadastro, consulta e atualização de leitos por setor e status
- Dashboard com KPIs de ocupação
- Previsão de data de alta com Random Forest
- Previsão de demanda de leitos para os próximos N dias
- API documentada automaticamente (Swagger UI)
- Gerador de dados sintéticos para demonstração e testes

---

## Stack

| Camada       | Tecnologia                          |
|--------------|-------------------------------------|
| Previsão     | Random Forest (scikit-learn)        |
| Backend      | Python + FastAPI                    |
| Banco (demo) | Em memória (pronto para PostgreSQL) |
| Dashboard    | Streamlit + Plotly                  |

---

## Estrutura do projeto

```text
smartleito/
├── backend/                 # API FastAPI
│   ├── app/
│   │   ├── main.py          # Entry point
│   │   ├── routers/         # /leitos e /previsao
│   │   ├── schemas/         # Pydantic models
│   │   └── services/        # Lógica de negócio + ML
│   └── requirements.txt
├── ml/
│   └── train_modelo_alta.py # Treina Random Forest
├── data/
│   └── gerar_dados_demo.py  # Gera CSVs de exemplo
├── dashboard/
│   ├── app.py               # Interface Streamlit
│   └── requirements.txt
└── README.md
```

---

## Como rodar

### Pré-requisitos

- [Git](https://git-scm.com/)
- [Python 3.10+](https://www.python.org/downloads/)

### 1. Clonar o repositório

```bash
git clone https://github.com/kenjiyuri/Projeto-Smarleito
cd smartleito
```

### 2. Backend (API)

Execute o backend primeiro, pois o dashboard consome seus dados.

```bash
cd backend

# (Recomendado) criar e ativar um ambiente virtual
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows

python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Swagger UI: http://localhost:8000/docs
- Mantenha este terminal aberto enquanto usa o dashboard.

### 3. Dashboard (Streamlit)

Em **outro terminal**, a partir da raiz do projeto:

```bash
cd dashboard
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

- O painel abre em http://localhost:8501
- Se a API não estiver rodando, o dashboard usa automaticamente **dados simulados locais**.

### 4. (Opcional) Treinar o modelo

Para recalibrar o Random Forest de previsão de alta:

```bash
cd ml
python -m pip install scikit-learn joblib numpy
python train_modelo_alta.py
```

O modelo é salvo em `ml/models/modelo_alta.joblib` e recarregado automaticamente pela API.

### 5. (Opcional) Gerar dados sintéticos

A partir da raiz do projeto:

```bash
python data/gerar_dados_demo.py
```

---

## Endpoints da API

Base URL: `http://localhost:8000/api/v1`

| Método | Rota                | Descrição                                 |
|--------|---------------------|-------------------------------------------|
| GET    | `/leitos`           | Lista leitos (filtros: `setor`, `status`) |
| GET    | `/leitos/resumo`    | KPIs do dashboard                         |
| GET    | `/leitos/{id}`      | Detalhe de um leito                       |
| POST   | `/leitos`           | Cria um leito                             |
| PATCH  | `/leitos/{id}`      | Atualiza status/setor                     |
| POST   | `/previsao/alta`    | Prevê a data de alta                      |
| POST   | `/previsao/demanda` | Prevê a demanda nos próximos N dias       |

### Exemplo: previsão de alta

```bash
curl -X POST http://localhost:8000/api/v1/previsao/alta \
  -H "Content-Type: application/json" \
  -d '{
    "idade": 67,
    "dias_internado": 4,
    "complexidade": 4,
    "setor": "UTI",
    "comorbidades": 2
  }'
```

---

## Solução de problemas

| Problema                          | Solução                                                         |
|-----------------------------------|-----------------------------------------------------------------|
| `pip` não é reconhecido (Windows) | Use `python -m pip ...`                                         |
| Porta 8000 ou 8501 em uso         | Troque a porta (`--port 8001` / `--server.port 8502`)           |
| Dashboard mostra dados simulados  | Verifique se a API está rodando em http://localhost:8000        |
| Erro ao carregar o modelo         | Execute o treinamento (passo 4) para gerar `modelo_alta.joblib` |


---

## Equipe

João Pedro · Kenji Yuri · Luca Casari · Lucia Maria · Matheus Bargas · Maria Luiza

**Unimar — Universidade de Marília**
