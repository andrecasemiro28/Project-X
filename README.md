# Project-X

# Flight Risk Predictor

**A tool that reveals hidden delay and cancellation risks before you book.**
Using accurate probability predictions, it helps passengers choose lower-risk flights and statistically reduce the financial and time costs of travel uncertainty.


# drive link
** https://drive.google.com/drive/folders/1LoptgYXrfqikYUDppOhGRmC-DARjlwAf?usp=drive_link **


## 🚀 Como Executar

### Método 1: Usando o ambiente virtual configurado
c:/Users/Juliano.jcs/dev/Project-X/.venv/Scripts/python.exe -m streamlit run app.py

### Método 2: Ativando o ambiente virtual primeiro
.\.venv\Scripts\activate
streamlit run app.py

### Método 3: Instalação em novo ambiente
#Criar ambiente virtual
python -m venv .venv

#Ativar ambiente
.\.venv\Scripts\activate

#Instalar dependências
pip install -r requirements.txt

#Executar aplicação
streamlit run app.py

## 📝 Observações Técnicas
1. Python 3.13 Compatibility: Inicialmente houve preocupação com compatibilidade, mas o google-generativeai instalou corretamente
2. Protobuf Version: Foi necessário downgrade de 6.33.4 para 5.29.5 (gerenciado automaticamente pelo pip)
3. Browser Preview: Simple Browser aberto em http://localhost:8501 para visualização da aplicação