# arquivo: modules/config.py
import os
from dotenv import load_dotenv
import pathlib

class Config:
    """Configurações da aplicação carregadas de variáveis de ambiente"""
    
    def __init__(self):
        # Certifica-se de que as variáveis de ambiente foram carregadas
        load_dotenv()
        
        # Diretório base do projeto
        self.base_dir = pathlib.Path(__file__).parent.parent.absolute()
        
        # Configurações do banco de dados
        self.db_uri = os.getenv('DATABASE_URI', 'postgresql://postgres:postgres@localhost:5432/dvdrental')
        
        # Configurações do OpenAI
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.model_name = os.getenv('MODEL_NAME', 'gpt-4')
        
        # Configurações dos arquivos de checkpoint
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.schema_checkpoint = os.path.join(self.data_dir, 'schema_checkpoint.pkl')
        self.vector_store_dir = os.path.join(self.data_dir, 'vector_store')
        
        # Certifica-se de que os diretórios necessários existem
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.vector_store_dir, exist_ok=True)