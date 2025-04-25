# arquivo: modules/sql_agent.py
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
from langchain_community.cache import SQLAlchemyCache
from langchain.globals import set_llm_cache
import datetime

# Constantes e configurações
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

# Tabelas por categoria
ODOO_TABLE_CATEGORIES = {
    "main_business": [
        "res_partner", "product_template", "product_product", "sale_order", "purchase_order",
        "account_move", "stock_move", "stock_quant", "res_users", "crm_lead", "hr_employee"
    ],
    "transactional": [
        "account_", "sale_", "purchase_", "stock_", "pos_", "mrp_", "project_"
    ],
    "configuration": [
        "res_company", "res_config_", "product_attribute", "uom_", "account_account",
        "account_journal"
    ],
    "technical": [
        "ir_", "base_", "bus_", "mail_", "auth_", "rule_", "workflow_"
    ],
    "log_or_history": [
        "_log", "_history", "_archive", "_report", "_dashboard", "_wizard"
    ]
}

# Mapa de prefixos para módulos
ODOO_MODULE_MAP = {
    "res": "base",
    "product": "product",
    "account": "accounting",
    "sale": "sales",
    "purchase": "purchase",
    "stock": "inventory",
    "mrp": "manufacturing",
    "hr": "human_resources",
    "crm": "customer_relationship",
    "pos": "point_of_sale",
    "project": "project_management"
}

class SchemaExtractor:
    """Classe aprimorada para extrair metadados do esquema do banco de dados"""
    
    def __init__(self, db_uri: str):
        self.db_uri = db_uri
        self.engine = create_engine(db_uri)
        self.inspector = inspect(self.engine)
        self.metadata = MetaData()
        self.is_odoo = self._check_if_odoo()
        self.table_category_cache = {}
        self.module_cache = {}
    
    def _check_if_odoo(self) -> bool:
        """Verifica se é um banco Odoo verificando tabelas características"""
        tables = self.get_all_tables()
        odoo_tables = {'ir_module_module', 'res_company', 'res_users'}
        return any(table in tables for table in odoo_tables)
    
    def _infer_odoo_module(self, table_name: str) -> str:
        """Infere o módulo Odoo com base no nome da tabela"""
        if table_name in self.module_cache:
            return self.module_cache[table_name]
        
        prefix = table_name.split('_')[0] if '_' in table_name else ''
        module = ODOO_MODULE_MAP.get(prefix, "other")
        
        # Verificações adicionais para casos especiais
        if table_name.startswith('ir_'):
            module = "base"
        elif table_name.startswith('base_'):
            module = "base"
        
        self.module_cache[table_name] = module
        return module
    
    def _identify_table_category(self, table_name: str) -> str:
        """Identifica a categoria da tabela baseada em seu nome"""
        if table_name in self.table_category_cache:
            return self.table_category_cache[table_name]
        
        # Verifica primeiramente as tabelas principais de negócio
        if table_name in ODOO_TABLE_CATEGORIES["main_business"]:
            category = "main_business"
        # Verifica se é tabela técnica
        elif any(table_name.startswith(prefix) for prefix in ODOO_TABLE_CATEGORIES["technical"]):
            category = "technical"
        # Verifica se é tabela de log ou histórico
        elif any(suffix in table_name for suffix in ODOO_TABLE_CATEGORIES["log_or_history"]):
            category = "log_or_history"
        # Verifica se é tabela de configuração
        elif any(table_name.startswith(prefix) for prefix in ODOO_TABLE_CATEGORIES["configuration"]):
            category = "configuration"
        # Verifica se é tabela transacional
        elif any(table_name.startswith(prefix) for prefix in ODOO_TABLE_CATEGORIES["transactional"]):
            category = "transactional"
        else:
            category = "other"
        
        self.table_category_cache[table_name] = category
        return category
    
    def _is_important_table(self, table_name: str) -> bool:
        """Determina se uma tabela é importante para consultas"""
        category = self._identify_table_category(table_name)
        
        # Tabelas principais de negócio e transacionais são importantes
        if category in ["main_business", "transactional"]:
            return True
        
        # Algumas tabelas de configuração são importantes
        if category == "configuration" and any(important in table_name 
                                            for important in ["company", "partner", "product"]):
            return True
        
        # Tabelas técnicas, logs e outras são menos importantes
        return False
    
    def _should_skip_table(self, table_name: str) -> bool:
        """Determina se uma tabela deve ser ignorada no processamento"""
        # Ignora tabelas problemáticas
        if table_name in PROBLEMATIC_TABLES and PROBLEMATIC_TABLES[table_name].get('skip', False):
            return True
        
        # Tabelas técnicas específicas para pular
        if table_name.startswith(('ir_attachment', 'ir_cron', 'ir_translation', 'ir_ui_view')):
            return True
        
        # Tabelas temporárias ou de sessão
        if '_tmp_' in table_name or table_name.startswith('session_'):
            return True
        
        # Tabelas de log ou histórico
        category = self._identify_table_category(table_name)
        if category == "log_or_history":
            # Excluindo a maioria das tabelas de log, mas mantendo algumas importantes
            if not any(important in table_name for important in ["audit", "accounting"]):
                return True
        
        return False
    
    def _should_include_sample_data(self, table_name: str) -> bool:
        """Determina se devemos incluir dados de amostra para uma tabela"""
        # Verifica configurações específicas para a tabela
        if table_name in PROBLEMATIC_TABLES:
            if PROBLEMATIC_TABLES[table_name].get('skip_sample_data', False):
                return False
        
        # Verifica configurações para tabelas complexas
        if table_name in ODOO_COMPLEX_TABLES:
            if ODOO_COMPLEX_TABLES[table_name].get('skip_sample_data', False):
                return False
        
        # Inclui amostrar apenas para tabelas importantes
        category = self._identify_table_category(table_name)
        return category in ["main_business", "transactional"]
    
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
        # Verifica se devemos pular índices para esta tabela
        if table_name in SPECIAL_TABLES and SPECIAL_TABLES[table_name].get('skip_indexes', False):
            return []
        
        try:
            return self.inspector.get_indexes(table_name)
        except Exception as e:
            print(f"Erro ao obter índices da tabela {table_name}: {str(e)}")
            return []
        
    def _json_serializable_converter(self, obj):
        """Converte tipos não serializáveis para formatos compatíveis com JSON"""
        import pandas as pd
        import numpy as np
        from datetime import datetime, date
    
        # Converte tipos de data/hora
        if isinstance(obj, (pd.Timestamp, datetime, date)):
            return obj.isoformat()
        # Converte tipos numpy
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # Converte decimal
        elif hasattr(obj, 'to_eng_string'):  # Para objetos Decimal
            return str(obj)
        # Outros tipos não serializáveis diretamente
        else:
            return str(obj)


    def get_table_sample_data(self, table_name: str, sample_size: int = 3) -> pd.DataFrame:
        """Retorna amostra de dados de uma tabela com tratamento para palavras reservadas"""
        # Verifica se devemos incluir amostras para esta tabela
        if not self._should_include_sample_data(table_name):
            return pd.DataFrame()
        
        if table_name in PROBLEMATIC_TABLES:
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
            "create_statement": self.get_table_create_statement(table_name),
            "category": self._identify_table_category(table_name),
            "module": self._infer_odoo_module(table_name),
            "indexes": self.get_table_indexes(table_name)
        }

        # Adiciona amostra de dados apenas para tabelas relevantes
        if self._should_include_sample_data(table_name):
            try:
                sample_data = self.get_table_sample_data(table_name)
                # Use o conversor personalizado ao transformar em dicionários
                if not sample_data.empty:
                    records = sample_data.to_dict(orient='records')
                    # Serialize cada registro com o conversor personalizado
                    table_info["sample_data"] = json.loads(json.dumps(records, default=self._json_serializable_converter))
                else:
                    table_info["sample_data"] = []
            except Exception as e:
                print(f"Erro ao obter amostra de dados da tabela {table_name}: {str(e)}")
                table_info["sample_data"] = []
        else:
            table_info["sample_data"] = []

        return table_info
    
    def format_table_info_for_embedding(self, table_info: Dict[str, Any]) -> str:
        """Formata informações da tabela para embedding"""
        table_name = table_info["table_name"]
        category = table_info.get("category", "unknown")
        module = table_info.get("module", "unknown")
        
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
CATEGORY: {category}
MODULE: {module}

COLUMNS:
{chr(10).join(column_info) if column_info else '-- Nenhuma coluna encontrada'}

FOREIGN KEYS:
{chr(10).join(fk_info) if fk_info else '-- Nenhuma chave estrangeira'}

INDEXES:
{chr(10).join(index_info) if index_info else '-- Nenhum índice definido'}

{sample_data_str}
"""
        return formatted_text
    
    def generate_module_info(self) -> List[Document]:
        """Gera documentos com informações sobre os módulos do Odoo"""
        if not self.is_odoo:
            return []
        
        # Agrupar tabelas por módulo
        module_tables = {}
        all_tables = self.get_all_tables()
        
        for table in all_tables:
            module = self._infer_odoo_module(table)
            if module not in module_tables:
                module_tables[module] = []
            module_tables[module].append(table)
        
        documents = []
        for module, tables in module_tables.items():
            # Conta tabelas por categoria neste módulo
            category_counts = {}
            for table in tables:
                category = self._identify_table_category(table)
                if category not in category_counts:
                    category_counts[category] = 0
                category_counts[category] += 1
            
            # Formata o conteúdo com informações do módulo
            content = f"""
MODULE: {module}
TABLES COUNT: {len(tables)}
CATEGORIES:
{chr(10).join([f"- {cat}: {count} tables" for cat, count in category_counts.items()])}

MAIN TABLES:
{chr(10).join([f"- {table}" for table in tables if self._is_important_table(table)][:10])}

ALL TABLES:
{', '.join(tables)}
"""
            
            doc = Document(
                page_content=content,
                metadata={
                    "type": "module", 
                    "module": module,
                    "table_count": len(tables)
                }
            )
            documents.append(doc)
        
        return documents
    
    def generate_schema_overview(self) -> Document:
        """Gera um documento de visão geral do esquema"""
        all_tables = self.get_all_tables()
        
        # Conta tabelas por categoria e módulo
        categories = {}
        modules = {}
        
        for table in all_tables:
            category = self._identify_table_category(table)
            module = self._infer_odoo_module(table)
            
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
            
            if module not in modules:
                modules[module] = 0
            modules[module] += 1
        
        content = f"""
DATABASE SCHEMA OVERVIEW
Total Tables: {len(all_tables)}
Database Type: {"Odoo" if self.is_odoo else "Generic SQL"}

CATEGORIES:
{chr(10).join([f"- {cat}: {count} tables" for cat, count in categories.items()])}

MODULES:
{chr(10).join([f"- {mod}: {count} tables" for mod, count in modules.items()])}

IMPORTANT BUSINESS TABLES:
{chr(10).join([f"- {table}" for table in all_tables if self._is_important_table(table)][:20])}
"""
        
        return Document(
            page_content=content,
            metadata={"type": "schema_overview"}
        )
    
    def extract_table_structure_document(self, table_name: str) -> Document:
        """Extrai documento apenas com a estrutura da tabela (sem amostras)"""
        # Obtém informações de estrutura da tabela
        columns = self.get_table_columns(table_name)
        primary_keys = self.get_primary_keys(table_name)
        foreign_keys = self.get_foreign_keys(table_name)
        category = self._identify_table_category(table_name)
        module = self._infer_odoo_module(table_name)
        
        # Formata as colunas
        column_info = []
        for col in columns:
            is_pk = col.get("name") in primary_keys
            pk_str = " (PRIMARY KEY)" if is_pk else ""
            col_str = f"- {col.get('name', 'Unknown')}: {str(col.get('type', 'Unknown'))}{pk_str}"
            column_info.append(col_str)
        
        # Formata as chaves estrangeiras
        fk_info = []
        for fk in foreign_keys:
            fk_cols = ", ".join(fk.get("constrained_columns", []) or [])
            ref_cols = ", ".join(fk.get("referred_columns", []) or [])
            ref_table = fk.get("referred_table", "Unknown")
            if fk_cols and ref_cols:
                fk_str = f"- {fk_cols} -> {ref_table}.{ref_cols}"
                fk_info.append(fk_str)
        
        # Formata o conteúdo
        content = f"""
TABLE: {table_name}
CATEGORY: {category}
MODULE: {module}

COLUMNS:
{chr(10).join(column_info) if column_info else '-- Nenhuma coluna encontrada'}

RELATIONSHIPS:
{chr(10).join(fk_info) if fk_info else '-- Nenhuma chave estrangeira'}
"""
        
        return Document(
            page_content=content,
            metadata={
                "table_name": table_name,
                "type": "table_structure",
                "category": category,
                "module": module
            }
        )
    
    # Em seguida, modifique a função extract_table_data_document para usar o conversor:

    def extract_table_data_document(self, table_name: str, sample_size: int = 3) -> Optional[Document]:
        """Extrai documento apenas com amostras de dados da tabela"""
        # Verifica se devemos incluir amostras para esta tabela
        if not self._should_include_sample_data(table_name):
            return None
    
        try:
            sample_data = self.get_table_sample_data(table_name, sample_size)
            if sample_data.empty:
                return None
        
            # Converte o DataFrame para dict e então para JSON usando o conversor personalizado
            records = sample_data.to_dict(orient='records')
        
            # Formata o conteúdo
            content = f"""
TABLE DATA SAMPLES: {table_name}

{chr(10).join([f"Row {i+1}: {json.dumps(row, default=self._json_serializable_converter)}" for i, row in enumerate(records)])}
"""
        
            return Document(
                page_content=content,
                metadata={
                    "table_name": table_name,
                    "type": "table_data",
                    "sample_count": len(records)
                }
            )
        except Exception as e:
            print(f"Erro ao extrair dados da tabela {table_name}: {str(e)}")
            return None
    
    def extract_all_tables_info(self) -> List[Document]:
        """Extrai informações de todas as tabelas com estratégia otimizada"""
        all_tables = self.get_all_tables()
        documents = []
        
        # Adiciona documento de visão geral
        documents.append(self.generate_schema_overview())
        
        # Adiciona documentos de módulos (se for Odoo)
        if self.is_odoo:
            module_docs = self.generate_module_info()
            documents.extend(module_docs)
        
        # Filtra tabelas por importância e processa primeiro as mais importantes
        important_tables = [t for t in all_tables if self._is_important_table(t)]
        other_tables = [t for t in all_tables if t not in important_tables and not self._should_skip_table(t)]
        
        print(f"Processando {len(important_tables)} tabelas importantes")
        for table in important_tables:
            print(f"Processando tabela importante: {table}")
            
            # Adiciona documento de estrutura
            structure_doc = self.extract_table_structure_document(table)
            documents.append(structure_doc)
            
            # Adiciona documento de dados (se aplicável)
            data_doc = self.extract_table_data_document(table)
            if data_doc:
                documents.append(data_doc)
        
        # Processa tabelas secundárias (apenas estrutura)
        print(f"Processando {len(other_tables)} tabelas secundárias")
        batch_size = 50
        for i in range(0, len(other_tables), batch_size):
            batch = other_tables[i:i+batch_size]
            for table in batch:
                print(f"Processando tabela secundária: {table}")
                structure_doc = self.extract_table_structure_document(table)
                documents.append(structure_doc)
        
        # Para tabelas que devem ser ignoradas, adiciona documento mínimo
        skipped_tables = [t for t in all_tables if self._should_skip_table(t)]
        print(f"Adicionando {len(skipped_tables)} tabelas ignoradas como referência mínima")
        
        for table in skipped_tables:
            minimal_doc = Document(
                page_content=f"TABLE: {table}\nNote: Esta é uma tabela técnica ou complexa que requer tratamento especial.",
                metadata={"table_name": table, "type": "minimal_reference"}
            )
            documents.append(minimal_doc)
        
        return documents

# --- Classe de Callback para Capturar a Query SQL ---
class SQLQueryCaptureCallback(BaseCallbackHandler):
    """Callback handler to capture the input query for SQL tools."""
    def __init__(self):
        super().__init__()
        self.sql_query: Optional[str] = None
        self.start_time: Optional[float] = None
        self.exec_time: Optional[float] = None

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Called when the tool starts running."""
        # Verifica se a ferramenta é uma das ferramentas SQL
        tool_name = serialized.get("name")
        if tool_name in ['sql_db_query', 'query-sql', 'sql_db_query_checker']:
            # Registra o tempo de início
            self.start_time = time.time()
            
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
    
    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Called when the tool finishes running."""
        if self.start_time is not None:
            self.exec_time = time.time() - self.start_time
            print(f"--- Callback: Tempo de execução da query: {self.exec_time:.2f}s ---")

    def get_captured_query(self) -> Optional[str]:
        """Returns the captured SQL query."""
        return self.sql_query
    
    def get_execution_time(self) -> Optional[float]:
        """Returns the query execution time if available."""
        return self.exec_time

    def reset(self):
        """Resets the captured query."""
        self.sql_query = None
        self.start_time = None
        self.exec_time = None

class OdooTextToSQL:
    """Classe principal otimizada para processamento de consultas text-to-SQL em bancos Odoo"""
    
    def __init__(self, db_uri: str, use_checkpoint: bool = True, force_reprocess: bool = False, 
                data_dir: Optional[str] = None, enable_llm_cache: bool = True):
        config = Config()
        self.db_uri = db_uri
        self.use_checkpoint = use_checkpoint
        self.force_reprocess = force_reprocess
        
        # Configuração do diretório de dados
        self.data_dir = data_dir or os.environ.get('SQL_AGENT_DATA_DIR', 'data')
        # Cria o diretório base se não existir
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.schema_extractor = SchemaExtractor(db_uri)
        self.embeddings = OpenAIEmbeddings(api_key=config.openai_api_key)

        # Configuração do cache SQLAlchemy para o LLM
        if enable_llm_cache:
            try:
                # Cria um engine SQLite para o cache no diretório de dados
                cache_path = os.path.join(self.data_dir, 'llm_cache.sqlite')
                cache_engine = create_engine(f"sqlite:///{cache_path}")
                
                # Configura o SQLAlchemyCache apenas com o engine
                self.llm_cache = SQLAlchemyCache(cache_engine)
                
                # Configura o cache global para LangChain
                set_llm_cache(self.llm_cache)
                print(f"Cache LLM SQLite configurado em: {cache_path}")
            except Exception as e:
                print(f"Erro ao configurar cache LLM: {str(e)}")

        # Inicialização do LLM
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
            prefix=SAFETY_PREFIX  # Usa a constante importada
        )
        # Instancia o callback handler
        self.query_callback_handler = SQLQueryCaptureCallback()
        # Cache para resultados de consultas
        self.query_cache = {}

    def _get_or_create_schema_embeddings(self) -> Chroma:
        """Cria ou recupera embeddings do schema do banco com estratégia otimizada"""
        try:
            schema_hash = self._generate_schema_hash()
            checkpoint_dir = os.path.join(self.data_dir, 'checkpoints', f"schema_{schema_hash}")
            
            # Cria o diretório de checkpoint se não existir
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Inicializa o client do Chroma
            embedding_function = self.embeddings
            
            if os.path.exists(os.path.join(checkpoint_dir, "chroma.sqlite3")) and not self.force_reprocess:
                print(f"Carregando embeddings do checkpoint: {checkpoint_dir}")
                return Chroma(
                    persist_directory=checkpoint_dir,
                    embedding_function=embedding_function
                )
            
            print(f"Gerando novos embeddings do schema em: {checkpoint_dir}")
            documents = self.schema_extractor.extract_all_tables_info()
            
            print(f"Total de documentos a serem incorporados: {len(documents)}")
            
            # Cria nova instância do Chroma
            vectorstore = Chroma(
                persist_directory=checkpoint_dir,
                embedding_function=embedding_function
            )
            
            # Adiciona documentos em lotes para melhor performance e feedback
            batch_size = 50
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                ids = [f"doc_{i+j}" for j in range(len(batch))]
                
                # Adiciona o lote com IDs explícitos
                vectorstore.add_documents(documents=batch, ids=ids)
                
                print(f"Processado lote {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} ({i+len(batch)}/{len(documents)} documentos)")
            
            print("Embeddings do schema concluídos e salvos.")
            return vectorstore
        except Exception as e:
            print(f"Erro ao criar embeddings: {str(e)}")
            # Fallback para diretório temporário em caso de erro
            fallback_dir = os.path.join(self.data_dir, 'checkpoints', 'fallback')
            os.makedirs(fallback_dir, exist_ok=True)
            print(f"Usando diretório de fallback: {fallback_dir}")
            return Chroma(persist_directory=fallback_dir, embedding_function=self.embeddings)

    def _generate_schema_hash(self) -> str:
        """Gera um hash representando o estado atual do schema com foco nas tabelas principais"""
        # Obtém lista de todas as tabelas
        all_tables = self.schema_extractor.get_all_tables()
        
        # Seleciona apenas tabelas importantes para o hash
        important_tables = [t for t in all_tables if self.schema_extractor._is_important_table(t)]
        
        # Para cada tabela importante, coleta informações essenciais para o hash
        schema_info = {}
        for table in important_tables:
            try:
                columns = self.schema_extractor.get_table_columns(table)
                primary_keys = self.schema_extractor.get_primary_keys(table)
                foreign_keys = self.schema_extractor.get_foreign_keys(table)
            
                # Simplifica para incluir apenas o essencial
                col_info = [(col["name"], str(col["type"])) for col in columns]
            
                schema_info[table] = {
                    "columns": col_info,
                    "primary_keys": primary_keys,
                    "foreign_keys": [(fk["constrained_columns"], fk["referred_table"]) for fk in foreign_keys]
                }
            except Exception as e:
                print(f"Erro ao processar tabela {table} para hash: {str(e)}")
                continue
    
        # Converte para string e gera hash
        schema_str = json.dumps(schema_info, sort_keys=True)
        return hashlib.md5(schema_str.encode()).hexdigest()
    
    def find_relevant_tables(self, query: str, top_k: int = 7) -> List[str]:
        """Encontra tabelas relevantes para a consulta com abordagem hierárquica"""
        # Primeiro, busca documentos de visão geral e módulos
        docs_overview = self.vector_store.similarity_search(
            query, 
            k=3,
            filter={"type": {"$in": ["schema_overview", "module"]}}
        )
        
        # Em seguida, busca documentos de estrutura de tabelas
        docs_structure = self.vector_store.similarity_search(
            query, 
            k=top_k,
            filter={"type": "table_structure"}
        )
        
        # Por fim, busca documentos de dados de tabelas
        docs_data = self.vector_store.similarity_search(
            query, 
            k=3,
            filter={"type": "table_data"}
        )
        
        # Extrai nomes de tabelas únicos de todos os resultados
        tables = []
        
        # Adiciona tabelas de documentos de estrutura (principal fonte)
        for doc in docs_structure:
            table_name = doc.metadata.get("table_name")
            if table_name and table_name not in tables:
                tables.append(table_name)
        
        # Adiciona tabelas de documentos de dados (complemento)
        for doc in docs_data:
            table_name = doc.metadata.get("table_name")
            if table_name and table_name not in tables:
                tables.append(table_name)
        
        # Se temos menos que 3 tabelas, busca documentos genéricos
        if len(tables) < 3:
            docs_generic = self.vector_store.similarity_search(query, k=top_k)
            for doc in docs_generic:
                table_name = doc.metadata.get("table_name")
                if table_name and table_name not in tables:
                    tables.append(table_name)
        
        return tables[:top_k]  # Limita ao número máximo de tabelas

    def get_table_relationships(self, tables: List[str]) -> Dict[str, List[str]]:
        """Identifica relacionamentos entre as tabelas relevantes"""
        relationships = {}
        
        for table in tables:
            related_tables = []
            
            # Busca chaves estrangeiras que saem desta tabela
            foreign_keys = self.schema_extractor.get_foreign_keys(table)
            for fk in foreign_keys:
                referred_table = fk.get("referred_table")
                if referred_table and referred_table in tables and referred_table != table:
                    related_tables.append(referred_table)
            
            # Busca tabelas que se referem a esta tabela
            for other_table in tables:
                if other_table == table:
                    continue
                    
                other_fks = self.schema_extractor.get_foreign_keys(other_table)
                for fk in other_fks:
                    if fk.get("referred_table") == table:
                        related_tables.append(other_table)
            
            relationships[table] = list(set(related_tables))  # Remove duplicatas
        
        return relationships

    def enhance_query_with_table_info(self, query: str, tables: List[str]) -> str:
        """Enriquece a consulta com informações sobre tabelas relevantes e suas relações"""
        # Coleta informações sobre estrutura das tabelas
        table_structures = []
        for table in tables:
            table_info = self.schema_extractor.extract_table_structure_document(table)
            table_structures.append(table_info.page_content)
        
        # Obtém relacionamentos entre as tabelas
        relationships = self.get_table_relationships(tables)
        relationship_info = []
        
        for table, related_tables in relationships.items():
            if related_tables:
                rel_str = f"Tabela {table} relaciona-se com: {', '.join(related_tables)}"
                relationship_info.append(rel_str)
        
        # Coleta amostras de dados apenas para tabelas principais (no máximo 3)
        sample_data_info = []
        important_tables = [t for t in tables if self.schema_extractor._is_important_table(t)][:3]
        
        for table in important_tables:
            data_doc = self.schema_extractor.extract_table_data_document(table)
            if data_doc:
                sample_data_info.append(data_doc.page_content)
        
        # Determina se é um banco Odoo para adicionar contexto específico
        odoo_context = ""
        if self.schema_extractor.is_odoo:
            odoo_context = """
Este é um banco de dados Odoo. Algumas dicas específicas para consultas Odoo:
- Tabelas res_partner contêm dados de parceiros/clientes
- Tabelas com prefixo account_ são relacionadas à contabilidade
- Tabelas com prefixo sale_ são relacionadas a vendas
- Tabelas com prefixo purchase_ são relacionadas a compras
- Tabelas com prefixo stock_ são relacionadas ao estoque
- Muitas tabelas usam campos 'active' para filtrar registros ativos (active=true)
- Campos como create_date, write_date são comuns para auditoria
"""
        
        # Cria prompt enriquecido
        enhanced_query = f"""
Pergunta: {query}

{odoo_context}

Tabelas relevantes para esta consulta:

{chr(10).join(table_structures)}

Relacionamentos entre tabelas:
{chr(10).join(relationship_info) if relationship_info else "Não foram identificados relacionamentos diretos entre estas tabelas."}

{chr(10).join(sample_data_info) if sample_data_info else ""}

Por favor, gere uma consulta SQL que responda à pergunta usando estas tabelas. 
Use joins quando apropriado e prefira JOINs explícitos (ex: INNER JOIN) em vez de junções na cláusula WHERE.
Inclua aliases de tabela para melhorar a legibilidade.
"""
        return enhanced_query
    
    def get_query_from_cache(self, question: str) -> Optional[Dict[str, Any]]:
        """Verifica se há resultado em cache para a pergunta"""
        # Usamos uma chave simplificada para maximizar chance de cache hit
        question_key = question.lower().strip()
        return self.query_cache.get(question_key)
    
    def add_query_to_cache(self, question: str, result: Dict[str, Any], query: Optional[str]):
        """Adiciona o resultado e query ao cache"""
        question_key = question.lower().strip()
        self.query_cache[question_key] = {
            'result': result,
            'query': query,
            'timestamp': time.time()
        }
    
    def query(self, user_question: str) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Processa pergunta do usuário, captura a query SQL via callback,
        e retorna o resultado do agente e a query capturada.
        """
        # Verifica primeiro o cache
        cached_data = self.get_query_from_cache(user_question)
        if cached_data:
            print("Usando resultado em cache")
            return cached_data['result'], cached_data['query']
        
        # Reseta o callback antes de cada consulta
        self.query_callback_handler.reset()
        
        # Encontra tabelas relevantes para a consulta
        relevant_tables = self.find_relevant_tables(user_question)
        print(f"Tabelas relevantes: {', '.join(relevant_tables)}")
        
        # Enriquece a pergunta com informações de schema
        enhanced_question = self.enhance_query_with_table_info(user_question, relevant_tables)
        
        # Registra tempo de início
        start_time = time.time()
        
        # Executa o agente passando o callback handler
        result = self.agent.invoke(
            {"input": enhanced_question},
            config={"callbacks": [self.query_callback_handler]} # Passa o handler aqui
        )
        
        # Registra tempo de execução total
        exec_time = time.time() - start_time
        print(f"Tempo total de processamento: {exec_time:.2f}s")
        
        # Obtém a query capturada pelo callback
        captured_query = self.query_callback_handler.get_captured_query()
        query_exec_time = self.query_callback_handler.get_execution_time()
        
        if query_exec_time:
            print(f"Tempo de execução da query: {query_exec_time:.2f}s")
        
        # Adiciona ao cache
        self.add_query_to_cache(user_question, result, captured_query)
        
        # Retorna tanto o resultado quanto a query capturada
        return result, captured_query            