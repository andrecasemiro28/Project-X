# Flight Risk AI 🛫

**Predição inteligente de riscos de atraso e cancelamento de voos**

Um aplicativo Streamlit que utiliza Machine Learning (XGBoost) e IA Generativa (Google Gemini) para analisar rotas de voo e fornecer recomendações personalizadas baseadas em 10 personas de viajantes.

## 🎯 Funcionalidades

- **Análise de Risco Preditiva**: Previsão de probabilidade de atraso baseada em padrões históricos de rotas
- **10 Personas de Viajantes**: Recomendações customizadas (Executivo, Estudante, Aposentado, Turista, Nômade Digital, etc.)
- **Chatbot IA Integrado**: Assistente conversacional com Google Gemini para orientações personalizadas
- **Análise de Custo de Oportunidade**: Cálculo do impacto financeiro de atrasos para cada persona
- **Cotação de Seguros**: Preços dinâmicos baseados no nível de risco
- **Filtros em Cascata**: Seleção intuitiva de Cidade → Aeroporto de Origem → Destino
- **391 Aeroportos Reais**: Base de dados completa com nomes de aeroportos dos EUA

## 🚀 Como Executar

### Pré-requisitos
- Python 3.13+
- Chave API do Google Gemini (configurada em `secrets.toml`)

### Método 1: Ambiente Virtual Configurado
```bash
c:/Users/Juliano.jcs/dev/Project-X/.venv/Scripts/python.exe -m streamlit run app.py
```

### Método 2: Ativando o Ambiente Virtual
```bash
# Windows
.\.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

# Executar aplicação
streamlit run app.py
```

### Método 3: Instalação do Zero
```bash
# Criar ambiente virtual
python -m venv .venv

# Ativar ambiente (Windows)
.\.venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt

# Executar aplicação
streamlit run app.py
```

## 🔑 Configuração da API Gemini

Crie um arquivo `secrets.toml` na raiz do projeto:

```toml
[gemini]
api_key = "sua-chave-api-aqui"
```

Obtenha sua chave em: https://aistudio.google.com/apikey

## 📊 Estrutura do Projeto

```
Project-X/
├── app.py                          # Aplicação principal Streamlit
├── data/
│   └── airports.json              # 391 aeroportos com nomes reais
├── flight_risk_app/
│   ├── flight_risk_model.json     # Modelo XGBoost treinado
│   ├── flight_risk_scaler.pkl     # StandardScaler
│   ├── flight_risk_encodings.pkl  # Label encodings
│   ├── flight_distance_lookup.pkl # Distâncias entre rotas
│   └── flight_traffic_stats.pkl   # Estatísticas de tráfego
├── scripts/
│   ├── extract_airports.py        # Extração de dados de aeroportos
│   ├── debug_cities.py            # Testes de filtros de cidade
│   ├── test_filters.py            # Validação de filtros
│   └── test_santa_barbara.py      # Testes específicos de rotas
├── requirements.txt               # Dependências Python
├── secrets.toml                   # Chave API Gemini (não versionado)
└── README.md                      # Este arquivo
```

## 🧠 Tecnologias Utilizadas

- **Streamlit 1.53.1**: Framework de UI
- **XGBoost 3.1.3**: Modelo de Machine Learning
- **scikit-learn 1.8.0**: Pré-processamento de dados
- **Google Generative AI**: Chatbot com Gemini 2.5 Flash
- **Plotly**: Visualizações interativas
- **Pandas**: Manipulação de dados

## 🎭 Personas Disponíveis

1. **Executive**: Alto valor de tempo, foco em produtividade
2. **Student**: Orçamento limitado, flexibilidade de agenda
3. **Parent**: Prioridade em previsibilidade e conforto familiar
4. **Retiree**: Valoriza conforto, baixa tolerância a estresse
5. **Tourist**: Busca experiências, médio orçamento
6. **Digital Nomad**: Alta flexibilidade, trabalha remotamente
7. **Explorer**: Aventureiro, tolerante a imprevistos
8. **VIP**: Máximo conforto, disposto a pagar por garantias
9. **Immigrant**: Viagens essenciais, sensível a custos
10. **Commuter**: Viagens frequentes, prioriza eficiência

## 📝 Observações Técnicas

- **Python 3.13**: Totalmente compatível (google-generativeai instalado com sucesso)
- **Protobuf**: Versão 5.29.5 (downgrade automático de 6.33.4)
- **sklearn**: Warning de versão (1.6.1 → 1.8.0) é não-bloqueante
- **Porta Padrão**: http://localhost:8501
- **Modo de Desenvolvimento**: Hot reload habilitado

## 🔗 Links Úteis

- **Drive do Projeto**: https://drive.google.com/drive/folders/1LoptgYXrfqikYUDppOhGRmC-DARjlwAf
- **Gemini API**: https://ai.google.dev/gemini-api/docs

## 📄 Licença

Projeto acadêmico/demonstrativo.