# Utiliza uma imagem leve do Python
FROM python:3.11-slim

# Define o diretório de trabalho
WORKDIR /app

# Copia apenas os arquivos de dependências primeiro para otimizar o cache
COPY requirements.txt ./

# Instala as dependências do sistema necessárias para psycopg2 e outros pacotes
RUN apt-get update && \
    apt-get install -y gcc libpq-dev && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    apt-get remove -y gcc && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Copia o restante do código da aplicação
COPY modules /app/
COPY app.py /app/

# Expõe a porta padrão do Streamlit
EXPOSE 8501

# Define a variável de ambiente para o Streamlit não abrir o navegador
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Comando para rodar a aplicação
CMD ["streamlit", "run", "app.py"]
