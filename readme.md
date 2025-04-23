# SQL Agent App

Uma aplicação web para converter perguntas em linguagem natural para consultas SQL usando LangChain e LLMs.

sql-agent-app/
│
├── .env                  # Arquivo com variáveis de ambiente (não versionado)
├── .env.example          # Exemplo de configuração do .env
├── app.py                # Ponto de entrada da aplicação Streamlit
├── requirements.txt      # Dependências do projeto
│
├── data/                 # Diretório para armazenar dados e checkpoints
│   ├── schema_checkpoint.pkl    # Checkpoint do schema do banco
│   └── vector_store/            # Diretório para o ChromaDB
│
└── modules/              # Módulos da aplicação
    ├── __init__.py       # Torna o diretório um pacote Python
    ├── config.py         # Configurações da aplicação
    └── sql_agent.py      # Implementação do agente SQL

## Características

- Interface web intuitiva com Streamlit
- Conversão de linguagem natural para SQL
- Suporte para banco de dados PostgreSQL
- Armazenamento eficiente de metadados do schema
- Sistema de cache para melhorar performance

## Requisitos

- Python 3.9+
- Banco de dados PostgreSQL (recomendado usar o exemplo DVD Rental)
- Chave de API da OpenAI

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/sql-agent-app.git
cd sql-agent-app
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Configure as variáveis de ambiente:
```bash
cp .env.example .env
# Edite o arquivo .env com suas configurações
```

4. Configure o banco de dados DVD Rental:
   - Baixe o arquivo DVD Rental de: https://www.postgresqltutorial.com/postgresql-getting-started/postgresql-sample-database/
   - Siga as instruções para importar o banco de dados

## Executando a aplicação

```bash
streamlit run app.py
```

Acesse a aplicação em http://localhost:8501

## Estrutura do projeto

- `app.py`: Ponto de entrada da aplicação Streamlit
- `modules/`: Módulos da aplicação
  - `config.py`: Gerenciamento de configurações
  - `sql_agent.py`: Implementação do agente SQL
- `data/`: Diretório para armazenar dados e checkpoints

## Como usar

1. Digite uma pergunta em linguagem natural sobre o banco de dados
2. Clique em "Executar consulta"
3. Veja a consulta SQL gerada e os resultados

## Extensibilidade

Para adaptar a outros bancos de dados:
1. Modifique a string de conexão no arquivo `.env`
2. Ajuste parâmetros como `top_k` para encontrar tabelas relevantes
3. Apague os checkpoints existentes para reconstruir os embeddings

## Contribuições

Contribuições são bem-vindas! Por favor, sinta-se à vontade para enviar pull requests.