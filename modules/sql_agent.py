# arquivo: modules/sql_agent.py
#from langchain_chroma import Chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# Importações corrigidas:
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, inspect, MetaData
from sqlalchemy.schema import CreateTable
from langchain.schema import Document
import os
import pickle
import pandas as pd
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple
from .config import Config
# Importações adicionadas para Callbacks
from langchain.callbacks.base import BaseCallbackHandler

class SchemaExtractor:
    """Classe para extrair metadados do esquema do banco de dados"""
    
    def __init__(self, db_uri: str):
        self.db_uri = db_uri
        self.engine = create_engine(db_uri)
        self.inspector = inspect(self.engine)
        self.metadata = MetaData()
    
    def get_all_tables(self) -> List[str]:
        """Retorna lista de todas as tabelas no banco de dados"""
        return self.inspector.get_table_names()
    
    def get_table_columns(self, table_name: str) -> List[Dict[str, Any]]:
        """Retorna informações sobre colunas de uma tabela"""
        return self.inspector.get_columns(table_name)
    
    def get_primary_keys(self, table_name: str) -> List[str]:
        """Retorna chaves primárias de uma tabela"""
        try:
            return self.inspector.get_pk_constraint(table_name)['constrained_columns']
        except:
            return []
    
    def get_foreign_keys(self, table_name: str) -> List[Dict[str, Any]]:
        """Retorna chaves estrangeiras de uma tabela"""
        return self.inspector.get_foreign_keys(table_name)
    
    def get_table_create_statement(self, table_name: str) -> str:
        """Retorna o comando CREATE TABLE para uma tabela"""
        try:
            # Carrega a tabela no metadata
            table = self.metadata.tables.get(table_name)
            if table is None:
                table = self.metadata.reflect(self.engine, only=[table_name])
                table = self.metadata.tables.get(table_name)
            
            if table:
                return str(CreateTable(table).compile(self.engine))
            return f"CREATE TABLE {table_name} (...)"
        except Exception as e:
            return f"CREATE TABLE {table_name} (...) -- Erro: {str(e)}"
    
    def get_table_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        """Retorna índices de uma tabela"""
        return self.inspector.get_indexes(table_name)
    
    def get_table_sample_data(self, table_name: str, sample_size: int = 3) -> pd.DataFrame:
        """Retorna amostra de dados de uma tabela"""
        try:
            query = f"SELECT * FROM {table_name} LIMIT {sample_size}"
            return pd.read_sql(query, self.engine)
        except:
            return pd.DataFrame()
    
    def generate_rich_table_info(self, table_name: str) -> Dict[str, Any]:
        """Gera informações detalhadas sobre uma tabela"""
        columns = self.get_table_columns(table_name)
        primary_keys = self.get_primary_keys(table_name)
        foreign_keys = self.get_foreign_keys(table_name)
        create_statement = self.get_table_create_statement(table_name)
        indexes = self.get_table_indexes(table_name)
        
        # Opcional: adicionar amostra de dados
        try:
            sample_data = self.get_table_sample_data(table_name)
            sample_data_dict = sample_data.to_dict(orient='records')
        except:
            sample_data_dict = []
        
        return {
            "table_name": table_name,
            "columns": columns,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys,
            "create_statement": create_statement,
            "indexes": indexes,
            "sample_data": sample_data_dict
        }
    
    def format_table_info_for_embedding(self, table_info: Dict[str, Any]) -> str:
        """Formata informações da tabela para embedding"""
        table_name = table_info["table_name"]
        
        # Formata informações sobre colunas
        column_info = []
        for col in table_info["columns"]:
            is_pk = col["name"] in table_info["primary_keys"]
            pk_str = " (PRIMARY KEY)" if is_pk else ""
            col_str = f"- {col['name']}: {col['type']}{pk_str}"
            column_info.append(col_str)
        
        # Formata informações sobre chaves estrangeiras
        fk_info = []
        for fk in table_info["foreign_keys"]:
            fk_cols = ", ".join(fk["constrained_columns"])
            ref_cols = ", ".join(fk["referred_columns"])
            fk_str = f"- Foreign Key: {fk_cols} -> {fk['referred_table']}.{ref_cols}"
            fk_info.append(fk_str)
        
        # Informações sobre índices
        index_info = []
        for idx in table_info["indexes"]:
            idx_cols = ", ".join(idx["column_names"])
            idx_type = "UNIQUE " if idx["unique"] else ""
            idx_str = f"- {idx_type}Index on ({idx_cols})"
            index_info.append(idx_str)
        
        # Amostra de dados (primeiras linhas)
        sample_data_str = ""
        if table_info["sample_data"]:
            sample_data_str = "Exemplos de dados:\n"
            for i, row in enumerate(table_info["sample_data"]):
                sample_data_str += f"Exemplo {i+1}: {row}\n"
        
        # Monta o documento final
        formatted_text = f"""
TABLE: {table_name}

CREATE STATEMENT:
{table_info['create_statement']}

COLUMNS:
{chr(10).join(column_info)}

FOREIGN KEYS:
{chr(10).join(fk_info) if fk_info else "Nenhuma chave estrangeira"}

INDEXES:
{chr(10).join(index_info) if index_info else "Nenhum índice definido"}

{sample_data_str}
"""
        return formatted_text
    
    def extract_all_tables_info(self) -> List[Document]:
        """Extrai informações de todas as tabelas e formata para LangChain"""
        tables = self.get_all_tables()
        documents = []
        
        for table in tables:
            print(f"Processando tabela: {table}")
            table_info = self.generate_rich_table_info(table)
            formatted_info = self.format_table_info_for_embedding(table_info)
            
            # Cria documento LangChain
            doc = Document(
                page_content=formatted_info,
                metadata={"table_name": table}
            )
            documents.append(doc)
        
        return documents

# --- Classe de Callback para Capturar a Query SQL ---
class SQLQueryCaptureCallback(BaseCallbackHandler):
    """Callback handler to capture the input query for SQL tools."""
    def __init__(self):
        super().__init__()
        self.sql_query: Optional[str] = None

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Called when the tool starts running."""
        # Verifica se a ferramenta é uma das ferramentas SQL
        tool_name = serialized.get("name")
        if tool_name in ['sql_db_query', 'query-sql', 'sql_db_query_checker']:
            # O input_str pode ser a query diretamente ou um JSON stringificado
            potential_query = None
            try:
                # Tenta decodificar como JSON se for um dict stringificado
                input_data = json.loads(input_str)
                if isinstance(input_data, dict) and 'query' in input_data:
                    potential_query = input_data['query']
            except json.JSONDecodeError:
                # Se não for JSON, assume que input_str é a query
                potential_query = input_str
            except TypeError:
                # Se input_str não for string-like
                potential_query = str(input_str) # Tenta converter

            # Armazena apenas se for uma query SELECT (para evitar armazenar resultados de checker)
            if potential_query and isinstance(potential_query, str) and "SELECT" in potential_query.upper():
                self.sql_query = potential_query
                print(f"--- Callback: Query SQL capturada: {self.sql_query} ---") # Log no console

    def get_captured_query(self) -> Optional[str]:
        """Returns the captured SQL query."""
        return self.sql_query

    def reset(self):
        """Resets the captured query."""
        self.sql_query = None

class DVDRentalTextToSQL:
    """Classe principal para processamento de consultas text-to-SQL"""
    
    def __init__(self, db_uri: str, use_checkpoint: bool = True, force_reprocess: bool = False):
        config = Config()
        self.db_uri = db_uri
        self.use_checkpoint = use_checkpoint
        self.force_reprocess = force_reprocess
        self.schema_extractor = SchemaExtractor(db_uri)
        self.embeddings = OpenAIEmbeddings(api_key=config.openai_api_key)
        self.llm = ChatOpenAI(
            temperature=0,
            model_name=config.model_name,
            api_key=config.openai_api_key
        )
        self.db = SQLDatabase.from_uri(db_uri)
        self.vector_store = self._get_or_create_schema_embeddings()
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)

        # REMOVER return_intermediate_steps=True, pois não funcionou
        self.agent = create_sql_agent(
            llm=self.llm,
            toolkit=self.toolkit,
            verbose=True,
            agent_type="openai-tools"
            # return_intermediate_steps=True # Removido
        )
        # Instancia o callback handler
        self.query_callback_handler = SQLQueryCaptureCallback()

    def _get_or_create_schema_embeddings(self) -> Chroma:
        """Carrega embeddings persistidos ou cria novos com detecção de mudanças"""
        config = Config()
    
        # Verifica se o diretório do vector store existe e contém dados
        vector_store_exists = os.path.exists(config.vector_store_dir) and len(os.listdir(config.vector_store_dir)) > 0
    
        #Arquivos para controle do schema
        schema_hash_file = os.path.join(config.data_dir, "schema_hash.txt")
        processed_flag = os.path.join(config.data_dir, "schema_processed.flag")
    
        # Gera um hash do schema atual (tabelas e estrutura)
        current_schema_hash = self._generate_schema_hash()
    
        # Verifica se o schema mudou desde o último processamento
        schema_changed = True
        if os.path.exists(schema_hash_file):
            with open(schema_hash_file, "r") as f:
                stored_hash = f.read().strip()
                schema_changed = stored_hash != current_schema_hash
    
        # Se o checkpoint existe e o schema não mudou, carrega do disco
        if (self.use_checkpoint and vector_store_exists and os.path.exists(processed_flag) 
                and not schema_changed and not self.force_reprocess):
            print("Carregando schema do vector store persistido...")
            # Carrega o vector store do disco
            vector_store = Chroma(
                persist_directory=config.vector_store_dir,
                embedding_function=self.embeddings
            )
            return vector_store
    
        # Se chegou aqui, precisa processar o schema novamente
        reason = ""
        if not vector_store_exists:
            reason = "vector store não existe"
        elif not os.path.exists(processed_flag):
            reason = "flag de processamento não encontrada"
        elif schema_changed:
            reason = "schema do banco foi modificado"
        elif self.force_reprocess:
            reason = "reprocessamento forçado"
    
        print(f"Extraindo schema do banco de dados... ({reason})")
        documents = self.schema_extractor.extract_all_tables_info()
    
        # Limpa o diretório do vector store se já existir
        if vector_store_exists:
            import shutil
            shutil.rmtree(config.vector_store_dir)
            os.makedirs(config.vector_store_dir, exist_ok=True)
    
        # Cria vector store e persiste no disco
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=config.vector_store_dir
        )
    
        # Salva o hash do schema atual
        with open(schema_hash_file, "w") as f:
            f.write(current_schema_hash)
    
        # Cria arquivo de flag indicando que o schema foi processado
        with open(processed_flag, "w") as f:
            f.write("Schema processed successfully")
    
        return vector_store

    def _generate_schema_hash(self) -> str:
        """Gera um hash representando o estado atual do schema"""
        # Obtém lista de tabelas
        tables = self.schema_extractor.get_all_tables()
    
        # Para cada tabela, coleta informações básicas de estrutura
        schema_info = {}
        for table in tables:
            columns = self.schema_extractor.get_table_columns(table)
            primary_keys = self.schema_extractor.get_primary_keys(table)
            foreign_keys = self.schema_extractor.get_foreign_keys(table)
        
            # Simplifica para incluir apenas o essencial para o hash
            col_info = [(col["name"], str(col["type"])) for col in columns]
        
            schema_info[table] = {
                "columns": col_info,
                "primary_keys": primary_keys,
                "foreign_keys": [(fk["constrained_columns"], fk["referred_table"]) for fk in foreign_keys]
            }
    
        # Converte para string e gera hash
        schema_str = json.dumps(schema_info, sort_keys=True)
        return hashlib.md5(schema_str.encode()).hexdigest()

    
    def find_relevant_tables(self, query: str, top_k: int = 5) -> List[str]:
        """Encontra tabelas relevantes para a consulta"""
        docs = self.vector_store.similarity_search(query, k=top_k)
        tables = [doc.metadata["table_name"] for doc in docs]
        return tables
    
    def enhance_query_with_table_info(self, query: str, tables: List[str]) -> str:
        """Enriquece a consulta com informações sobre tabelas relevantes"""
        # Coleta informações detalhadas sobre as tabelas relevantes
        table_infos = []
        for table in tables:
            table_info = self.schema_extractor.generate_rich_table_info(table)
            formatted_info = self.schema_extractor.format_table_info_for_embedding(table_info)
            table_infos.append(formatted_info)
        
        # Cria prompt enriquecido
        enhanced_query = f"""
Pergunta: {query}

Tabelas relevantes para esta consulta:

{chr(10).join(table_infos)}

Por favor, gere uma consulta SQL que responda à pergunta usando estas tabelas. 
Use joins quando apropriado e prefira JOINs explícitos (ex: INNER JOIN) em vez de junções na cláusula WHERE.
"""
        return enhanced_query
    
    def query(self, user_question: str) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Processa pergunta do usuário, captura a query SQL via callback,
        e retorna o resultado do agente e a query capturada.
        """
        # Reseta o callback antes de cada consulta
        self.query_callback_handler.reset()

        relevant_tables = self.find_relevant_tables(user_question)
        print(f"Tabelas relevantes: {', '.join(relevant_tables)}")
        enhanced_question = self.enhance_query_with_table_info(user_question, relevant_tables)

        # Executa o agente passando o callback handler
        result = self.agent.invoke(
            {"input": enhanced_question},
            config={"callbacks": [self.query_callback_handler]} # Passa o handler aqui
        )

        # Obtém a query capturada pelo callback
        captured_query = self.query_callback_handler.get_captured_query()

        # Retorna tanto o resultado quanto a query capturada
        return result, captured_query