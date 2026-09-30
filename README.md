# SMARTLEITO

**Gestão inteligente de leitos hospitalares**

Projeto baseado no pitch acadêmico (Unimar). Prevê demanda de leitos, data de alta e disponibiliza um dashboard de gestão.

---

## Stack

| Camada        | Tecnologia                          |
|---------------|-------------------------------------|
| Previsão      | Random Forest (scikit-learn)        |
| Backend       | Python + FastAPI                    |
| Banco (demo)  | Em memória (pronto para PostgreSQL) |
| Dashboard     | Streamlit + Plotly                  |

---

## Estrutura

```
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

### 1. Backend (API)

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Documentação interativa: http://localhost:8000/docs

### 2. (Opcional) Treinar o modelo de alta

```bash
cd ml
pip install scikit-learn joblib numpy
python train_modelo_alta.py
```

O modelo é salvo em `ml/models/modelo_alta.joblib` e carregado automaticamente pela API.

### 3. Dashboard

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

Abre em http://localhost:8501  
Se a API estiver rodando, o dashboard consome os endpoints; senão, usa dados locais.

### 4. Gerar CSVs de demonstração

```bash
python data/gerar_dados_demo.py
```

---

## Endpoints principais

| Método | Rota                        | Descrição                          |
|--------|-----------------------------|------------------------------------|
| GET    | `/api/v1/leitos`            | Lista leitos (filtros: setor, status) |
| GET    | `/api/v1/leitos/resumo`     | KPIs do dashboard                  |
| GET    | `/api/v1/leitos/{id}`       | Detalhe de um leito                |
| POST   | `/api/v1/leitos`            | Cria leito                         |
| PATCH  | `/api/v1/leitos/{id}`       | Atualiza status/setor              |
| POST   | `/api/v1/previsao/alta`     | Prevê data de alta                 |
| POST   | `/api/v1/previsao/demanda`  | Prevê demanda nos próximos N dias  |

### Exemplo — previsão de alta

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

## Equipe

João Pedro · Kenji Yuri · Luca Casari · Lucia Maria · Matheus Bargas · Maria Luiza  

**Unimar — Universidade de Marília**
