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
# --- Importa o prompt do arquivo prompts.py ---
from .prompts import SAFETY_PREFIX
import time

ODOO_COMPLEX_TABLES = {
    'ir_filters': {
        'skip_sample_data': True,
        'max_column_length': 100  # limitar tamanho das colunas serializada
    },
    'ir_actions': {
        'skip_sample_data': True
    },
    'ir_ui_view': {
        'skip_sample_data': True
    },
    'ir_translation': {
        'skip_sample_data': True
    }
}

PROBLEMATIC_TABLES = {
    'ir_filters': {'skip': True},
    'ir_translation': {'skip': True},
    'ir_ui_view': {'skip': True},
    'table_privileges': {
        'skip_sample_data': True,
        'reserved_words': ['select', 'update', 'insert', 'delete']
    },
    'timeline': {
        'skip_sample_data': True,
        'reserved_words': ['default']
    },
    'bve_view_line': {
        'skip_sample_data': True,
        'reserved_words': ['column']
    },
    'core_user': {'skip': True},
    'metabase_field': {'skip': True},
    'view_log': {'skip': True},
    'audit_log': {'skip': True}
}

SPECIAL_TABLES = {
    'l10n_br_fiscal_dfe': {'skip_indexes': True},
    'qrtz_calendars': {'skip_indexes': True},
    'nfe_40_fordia': {'skip_indexes': True},
    'metabase_field': {'skip_indexes': True},
    'query': {'skip_indexes': True}
}

class SchemaExtractor:
    """Classe para extrair metadados do esquema do banco de dados"""
    
    def __init__(self, db_uri: str):
        self.db_uri = db_uri
        self.engine = create_engine(db_uri)
        self.inspector = inspect(self.engine)
        self.metadata = MetaData()
        self.is_odoo = self._check_if_odoo()
    
    def _check_if_odoo(self) -> bool:
        """Verifica se é um banco Odoo verificando tabelas características"""
        tables = self.get_all_tables()
        odoo_tables = {'ir_module_module', 'res_company', 'res_users'}
        return any(table in tables for table in odoo_tables)
    
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
        """Retorna amostra de dados de uma tabela com tratamento para palavras reservadas"""
        if table_name in PROBLEMATIC_TABLES:
            if PROBLEMATIC_TABLES[table_name].get('skip', False):
                return pd.DataFrame()
            
            if PROBLEMATIC_TABLES[table_name].get('skip_sample_data', False):
                return pd.DataFrame()
            
            # Trata palavras reservadas
            reserved_words = PROBLEMATIC_TABLES[table_name].get('reserved_words', [])
            if reserved_words:
                try:
                    columns = self.get_table_columns(table_name)
                    safe_columns = [
                        f'"{col["name"]}"' if col["name"] in reserved_words else col["name"]
                        for col in columns
                    ]
                    cols = ', '.join(safe_columns)
                    query = f"SELECT {cols} FROM {table_name} LIMIT {sample_size}"
                    return pd.read_sql(query, self.engine)
                except Exception as e:
                    print(f"Erro ao obter amostra da tabela {table_name}: {str(e)}")
                    return pd.DataFrame()
        
        try:
            query = f"SELECT * FROM {table_name} LIMIT {sample_size}"
            return pd.read_sql(query, self.engine)
        except Exception as e:
            print(f"Erro ao obter amostra da tabela {table_name}: {str(e)}")
            return pd.DataFrame()
    
    def generate_rich_table_info(self, table_name: str) -> Dict[str, Any]:
        """Gera informações detalhadas sobre uma tabela com tratamento especial"""
        table_info = {
            "table_name": table_name,
            "columns": self.get_table_columns(table_name),
            "primary_keys": self.get_primary_keys(table_name),
            "foreign_keys": self.get_foreign_keys(table_name),
            "create_statement": self.get_table_create_statement(table_name)
        }
        
        # Só adiciona índices se a tabela não estiver na lista de tabelas especiais
        # ou se não tiver a flag skip_indexes
        if not (table_name in SPECIAL_TABLES and SPECIAL_TABLES[table_name].get('skip_indexes', False)):
            try:
                table_info["indexes"] = self.get_table_indexes(table_name)
            except Exception as e:
                print(f"Erro ao obter índices da tabela {table_name}: {str(e)}")
                table_info["indexes"] = []
        else:
            table_info["indexes"] = []
        
        # Tenta obter amostra de dados
        try:
            sample_data = self.get_table_sample_data(table_name)
            table_info["sample_data"] = sample_data.to_dict(orient='records') if not sample_data.empty else []
        except Exception as e:
            print(f"Erro ao obter amostra de dados da tabela {table_name}: {str(e)}")
            table_info["sample_data"] = []
        
        return table_info
    
    def format_table_info_for_embedding(self, table_info: Dict[str, Any]) -> str:
        """Formata informações da tabela para embedding com tratamento para valores None"""
        table_name = table_info["table_name"]
        
        # Formata informações sobre colunas
        column_info = []
        for col in table_info.get("columns", []):
            if not isinstance(col, dict):
                continue
            is_pk = col.get("name") in table_info.get("primary_keys", [])
            pk_str = " (PRIMARY KEY)" if is_pk else ""
            col_str = f"- {col.get('name', 'Unknown')}: {col.get('type', 'Unknown')}{pk_str}"
            column_info.append(col_str)
        
        # Formata informações sobre chaves estrangeiras
        fk_info = []
        for fk in table_info.get("foreign_keys", []):
            if not isinstance(fk, dict):
                continue
            fk_cols = ", ".join(fk.get("constrained_columns", []) or [])
            ref_cols = ", ".join(fk.get("referred_columns", []) or [])
            ref_table = fk.get("referred_table", "Unknown")
            if fk_cols and ref_cols:
                fk_str = f"- Foreign Key: {fk_cols} -> {ref_table}.{ref_cols}"
                fk_info.append(fk_str)
        
        # Informações sobre índices com tratamento para None
        index_info = []
        for idx in table_info.get("indexes", []):
            if not isinstance(idx, dict):
                continue
            
            # Trata column_names None ou vazios
            column_names = idx.get("column_names", [])
            if column_names is None:
                column_names = []
            elif isinstance(column_names, str):
                column_names = [column_names]
            
            # Filtra apenas nomes de colunas válidos
            valid_columns = [str(col) for col in column_names if col is not None]
            
            if valid_columns:  # Só adiciona o índice se tiver colunas válidas
                idx_cols = ", ".join(valid_columns)
                idx_type = "UNIQUE " if idx.get("unique", False) else ""
                idx_str = f"- {idx_type}Index on ({idx_cols})"
                index_info.append(idx_str)
        
        # Amostra de dados (primeiras linhas)
        sample_data_str = ""
        sample_data = table_info.get("sample_data", [])
        if sample_data:
            sample_data_str = "Exemplos de dados:\n"
            for i, row in enumerate(sample_data):
                if isinstance(row, dict):
                    sample_data_str += f"Exemplo {i+1}: {row}\n"
        
        # Monta o documento final com verificações de existência
        formatted_text = f"""
TABLE: {table_name}

CREATE STATEMENT:
{table_info.get('create_statement', '-- Create statement não disponível')}

COLUMNS:
{chr(10).join(column_info) if column_info else '-- Nenhuma coluna encontrada'}

FOREIGN KEYS:
{chr(10).join(fk_info) if fk_info else '-- Nenhuma chave estrangeira'}

INDEXES:
{chr(10).join(index_info) if index_info else '-- Nenhum índice definido'}

{sample_data_str}
"""
        return formatted_text
    
    def extract_all_tables_info(self) -> List[Document]:
        """Extrai informações de todas as tabelas com melhor tratamento de erros"""
        tables = self.get_all_tables()
        documents = []
        batch_size = 50
        
        # Move tabelas problemáticas para o final
        tables = sorted(tables, key=lambda x: x in PROBLEMATIC_TABLES)
        
        for i in range(0, len(tables), batch_size):
            batch_tables = tables[i:i+batch_size]
            for table in batch_tables:
                print(f"Processando tabela {i+len(documents)+1}/{len(tables)}: {table}")
                
                # Se é uma tabela para pular, cria um documento simplificado
                if table in PROBLEMATIC_TABLES and PROBLEMATIC_TABLES[table].get('skip', False):
                    simple_info = f"""
                    TABLE: {table}
                    NOTICE: Esta é uma tabela complexa que requer tratamento especial.
                    Para consultas detalhadas desta tabela, por favor use queries SQL diretas.
                    """
                    doc = Document(
                        page_content=simple_info,
                        metadata={"table_name": table, "is_complex": True}
                    )
                    documents.append(doc)
                    continue
                
                try:
                    table_info = self.generate_rich_table_info(table)
                    formatted_info = self.format_table_info_for_embedding(table_info)
                    
                    doc = Document(
                        page_content=formatted_info,
                        metadata={"table_name": table}
                    )
                    documents.append(doc)
                except Exception as e:
                    print(f"Erro ao processar tabela {table}: {str(e)}")
                    continue
            
            print(f"Completado lote de {len(batch_tables)} tabelas")
        
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

        # --- Usa o prefixo importado ---
        self.agent = create_sql_agent(
            llm=self.llm,
            toolkit=self.toolkit,
            verbose=True,
            agent_type="openai-tools",
            prefix=SAFETY_PREFIX # Usa a constante importada
        )
        # Instancia o callback handler
        self.query_callback_handler = SQLQueryCaptureCallback()

    def _get_or_create_schema_embeddings(self) -> Chroma:
        """Cria ou recupera embeddings do schema do banco"""
        schema_hash = self._generate_schema_hash()
        checkpoint_dir = f"checkpoints/schema_{schema_hash}"
        
        # Cria o diretório de checkpoint se não existir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Inicializa o client do Chroma
        embedding_function = OpenAIEmbeddings()
        
        if os.path.exists(os.path.join(checkpoint_dir, "chroma.sqlite3")) and not self.force_reprocess:
            print("Carregando embeddings do checkpoint...")
            return Chroma(
                persist_directory=checkpoint_dir,
                embedding_function=embedding_function
            )
        
        print("Gerando novos embeddings do schema...")
        extractor = SchemaExtractor(self.db_uri)
        documents = extractor.extract_all_tables_info()
        
        # Cria nova instância do Chroma
        vectorstore = Chroma(
            persist_directory=checkpoint_dir,
            embedding_function=embedding_function
        )
        
        # Adiciona documentos em lotes
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            vectorstore.add_documents(batch)
            # O Chroma mais recente persiste automaticamente quando persist_directory é fornecido
            print(f"Adicionado lote de embeddings {i+1}/{len(documents)}")
        
        return vectorstore

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