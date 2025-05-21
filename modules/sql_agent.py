# arquivo: modules/sql_agent.py
from langchain_chroma import Chroma
from typing import Optional, Dict, Any, List, Tuple, Union, Pattern, Match, Callable
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
from .config import Config
# Importações adicionadas para Callbacks
from langchain.callbacks.base import BaseCallbackHandler
# --- Importa os prompts do arquivo prompts.py ---
from .prompts import SAFETY_PREFIX, QUERY_CHECKER
from .examples import FEW_SHOT_EXAMPLES
from .context import ODOO_CONTEXT
import time
from langchain_community.cache import SQLAlchemyCache
from langchain.globals import set_llm_cache
import datetime
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

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
    """Callback handler para capturar e interceptar consultas SQL.
    
    Este handler é capaz de capturar consultas SQL geradas pelo agente e potencialmente
    substituí-las por consultas adaptadas de exemplos conhecidos, permitindo maior
    precisão e controle sobre as consultas executadas no banco de dados.
    """
    def __init__(self):
        super().__init__()
        self.sql_query: Optional[str] = None
        self.start_time: Optional[float] = None
        self.exec_time: Optional[float] = None
        self.user_question: Optional[str] = None
        # Flags para controle de exemplos
        self.example_similarity_computed: bool = False
        self.usando_exemplo_adaptado: bool = False
        # Configurações para adaptação genérica
        self.generic_adaptations: List[Dict[str, Any]] = [
            # Adaptação de períodos em dias
            {
                "name": "interval_days",
                "sql_pattern": r"INTERVAL\s+'(\d+)\s+days'",
                "question_pattern": r"\b(\d+)\s*dias\b",
                "replace_template": "'{}' days'"
            },
            # Adaptação de anos
            {
                "name": "year",
                "sql_pattern": r"EXTRACT\s*\(\s*YEAR\s+FROM\s+[^\)]+\)\s*=\s*(\d{4})",
                "question_pattern": r"\b(19|20\d{2})\b", 
                "replace_template": "= {}"
            },
            # Adaptação de limites (LIMIT)
            {
                "name": "limit",
                "sql_pattern": r"LIMIT\s+(\d+)",
                "question_pattern": r"\b(\d+)\s*produtos\b",
                "replace_template": "LIMIT {}"
            },
            # Adaptação de TOP N
            {
                "name": "top_n",
                "sql_pattern": r"TOP\s+(\d+)",
                "question_pattern": r"\b(\d+)\s*(primeiros|principais|melhores)\b",
                "replace_template": "TOP {}"
            },
            # Adaptação de períodos em meses
            {
                "name": "interval_months",
                "sql_pattern": r"INTERVAL\s+'(\d+)\s+month(s)?'",
                "question_pattern": r"\b(\d+)\s*m[êe]s(es)?\b",
                "replace_template": "'{}' month'"
            }
        ]
    
    def set_user_question(self, question: str):
        """Define a pergunta do usuário para possível análise de exemplos."""
        self.user_question = question
        self.example_similarity_computed = False
    
    def apply_generic_adaptations(self, query: str, user_question: str) -> str:
        """Aplica adaptações genéricas à consulta com base na pergunta do usuário"""
        # Por enquanto, apenas retorna a consulta original
        # Este método será expandido conforme novas adaptações forem identificadas
        return query

    def execute_query_with_optimized_context(self, user_question: str) -> Dict[str, Any]:
        """
        Executa a consulta SQL com contexto otimizado para bancos Odoo grandes.
        
        Esta função implementa o fluxo completo de consulta otimizada:
        1. Identifica tabelas relevantes para a pergunta
        2. Carrega apenas os esquemas dessas tabelas
        3. Executa a consulta com contexto reduzido e otimizações Odoo
        
        Args:
            user_question: Pergunta do usuário em linguagem natural
            
        Returns:
            Dicionário com resultados da consulta e metadados
        """
        # Inicializar métricas
        metrics = {
            "timestamp": datetime.datetime.now().isoformat(),
            "user_question": user_question,
            "cache_hit": False,
            "error": False,
            "request_id": hashlib.md5(f"{user_question}_{time.time()}".encode()).hexdigest(),
        }
        
        start_time = time.time()
        
        # Verificar cache primeiro (se habilitado)
        cache_key = self._generate_cache_key(user_question)
        cached_result = self.get_from_cache(cache_key)
        if cached_result:
            print(f"Cache hit para a consulta: {user_question[:30]}...")
            metrics["cache_hit"] = True
            metrics["execution_time_ms"] = 0
            
            # Adicionar métricas ao resultado
            cached_result["metrics"] = metrics
            return cached_result
        
        try:
            # 1. Identificar tabelas relevantes (reutilizando método existente)
            relevant_tables = self.find_relevant_tables(user_question)
            print(f"Tabelas relevantes identificadas: {relevant_tables}")
            metrics["tables_identified"] = relevant_tables
            metrics["tables_count"] = len(relevant_tables)
            
            # 2. Melhorar tabelas com relações Odoo importantes
            enhanced_tables = self._enhance_with_odoo_relations(relevant_tables)
            print(f"Tabelas relevantes após otimização: {enhanced_tables}")
            metrics["tables_enhanced"] = enhanced_tables
            metrics["tables_enhanced_count"] = len(enhanced_tables)
            
            # 3. Construir contexto otimizado para a consulta
            context_start = time.time()
            query_with_context = self._build_optimized_context(user_question, enhanced_tables)
            metrics["context_build_time_ms"] = round((time.time() - context_start) * 1000)
            
            # 4. Executar a consulta com timeout de segurança
            execution_start = time.time()
            
            # Configurar o tamanho do lote conforme configurações
            batch_size = 1000  # Valor padrão
            if hasattr(self, 'config') and hasattr(self.config, 'odoo_optimizations'):
                batch_size = getattr(self.config.odoo_optimizations, 'batch_size', 1000)
            
            # Executar com limite de tempo
            query_timeout = 30  # Valor padrão
            if hasattr(self, 'config') and hasattr(self.config, 'performance'):
                query_timeout = getattr(self.config.performance, 'query_timeout', 30)
                
            result = self._execute_with_timeout(
                query_with_context,
                timeout_seconds=query_timeout
            )
            
            query_time = time.time() - execution_start
            total_time = time.time() - start_time
            
            # Registrar métricas completas
            metrics.update({
                "total_execution_time_ms": round(total_time * 1000),
                "query_execution_time_ms": round(query_time * 1000),
                "success": result.get("success", False)
            })
            
            # Capturar a consulta SQL gerada (se disponível)
            if "sql_query" in result:
                metrics["sql_query"] = result["sql_query"]
                
                # Contar o número de consultas SQL geradas
                metrics["sql_queries_count"] = 1
                
                # Capturar tabelas utilizadas na consulta final
                used_tables = self._extract_tables_from_query(result["sql_query"])
                metrics["tables_used_in_query"] = used_tables
                metrics["tables_used_count"] = len(used_tables)
                
                # Verificar eficácia da identificação de tabelas
                if used_tables:
                    identification_accuracy = len(set(used_tables).intersection(set(enhanced_tables))) / len(used_tables)
                    metrics["table_identification_accuracy"] = round(identification_accuracy, 2)
            
            # 5. Cachear resultado se for bem sucedido e rápido o suficiente
            if result.get("success", False):
                max_query_time = 10  # Valor padrão
                cache_ttl = 3600  # Valor padrão (1 hora)
                
                if hasattr(self, 'config') and hasattr(self.config, 'performance'):
                    max_query_time = getattr(self.config.performance, 'max_query_execution_time', 10)
                    cache_ttl = getattr(self.config.performance, 'cache_ttl', 3600)
                
                if query_time < max_query_time and cache_ttl > 0:
                    self.add_to_cache(cache_key, result, ttl=cache_ttl)
            
            # Adicionar métricas ao resultado
            result["metrics"] = metrics
            return result
            
        except Exception as e:
            error_message = str(e)
            print(f"Erro ao executar consulta: {error_message}")
            
            # Registrar erro nas métricas
            metrics.update({
                "error": True,
                "error_message": error_message,
                "total_execution_time_ms": round((time.time() - start_time) * 1000)
            })
            
            return {
                "success": False,
                "error": error_message,
                "metrics": metrics
            }
        
    def _enhance_with_odoo_relations(self, table_names: List[str]) -> List[str]:
        """
        Amplia a lista de tabelas incluindo relações importantes do Odoo.
        
        Args:
            table_names: Lista inicial de nomes de tabelas
            
        Returns:
            Lista ampliada incluindo tabelas relacionadas importantes
        """
        if not table_names:
            return []
            
        enhanced_tables = table_names.copy()
        
        # Mapear tabelas Odoo que devem ser incluídas juntas (tabelas fundamentais)
        related_tables = {
            "sale_order": ["sale_order_line", "res_partner"],
            "purchase_order": ["purchase_order_line", "res_partner"],
            "product_product": ["product_template", "product_category"],
            "account_move": ["account_move_line", "res_partner"],
            "stock_move": ["stock_location", "product_product"],
            "res_partner": ["res_partner_bank", "res_company"],
        }
        
        # Adicionar tabelas relacionadas para cada tabela encontrada
        for table in table_names:
            if table in related_tables:
                for related in related_tables[table]:
                    if related not in enhanced_tables:
                        enhanced_tables.append(related)
        
        # Garantir que tabelas básicas importantes estejam presentes
        core_tables = ["res_company", "res_currency"]
        for core in core_tables:
            if core not in enhanced_tables:
                enhanced_tables.append(core)
        
        # Verificar relacionamentos no banco usando o método existente
        try:
            relationships = self.get_table_relationships(table_names)
            for table, related in relationships.items():
                for rel_table in related:
                    if rel_table not in enhanced_tables:
                        # Verificar se é uma tabela importante (não técnica)
                        if not rel_table.startswith("ir_") and self.schema_extractor._is_important_table(rel_table):
                            enhanced_tables.append(rel_table)
        except Exception as e:
            print(f"Aviso: Erro ao obter relacionamentos de tabelas: {str(e)}")
        
        # Limitar o número total de tabelas para evitar tokens excessivos
        max_total_tables = 15  # Ajustado para cobrir mais relacionamentos
        if len(enhanced_tables) > max_total_tables:
            # Priorizar tabelas originais e core tables
            priority_tables = table_names + [t for t in core_tables if t not in table_names]
            # Completar com outras tabelas relacionadas até o limite
            remaining = [t for t in enhanced_tables if t not in priority_tables]
            enhanced_tables = priority_tables + remaining[:max_total_tables-len(priority_tables)]
        
        return enhanced_tables
    
    def _build_optimized_context(self, user_question: str, tables: List[str]) -> str:
        """
        Constrói um contexto otimizado com base nas tabelas relevantes.
        
        Args:
            user_question: Pergunta do usuário
            tables: Lista de tabelas relevantes
            
        Returns:
            Consulta enriquecida com contexto de esquema otimizado
        """
        # Reutilizar o método existente enhance_query_with_table_info para esquemas
        enhanced_query = self.enhance_query_with_table_info(user_question, tables)
        
        # Adicionar contexto específico do Odoo para melhor compreensão
        if hasattr(self, 'odoo_context') and self.odoo_context:
            enhanced_query = f"{enhanced_query}\n\nContexto do Odoo: {self.odoo_context}"
        else:
            # Usar o contexto importado
            from .context import ODOO_CONTEXT
            enhanced_query = f"{enhanced_query}\n\nContexto do Odoo: {ODOO_CONTEXT[:500]}..."
        
        # Adicionar dicas de otimização para o LLM
        optimization_hints = """
        Dicas de otimização para consultas SQL no Odoo:
        1. Use CTEs (WITH) para consultas complexas
        2. Limite os resultados usando LIMIT quando possível
        3. Evite joins desnecessários
        4. Utilize índices disponíveis nas tabelas
        5. Filtre cedo no pipeline da consulta
        """
        
        enhanced_query = f"{enhanced_query}\n\n{optimization_hints}"
        
        return enhanced_query
    
    def _execute_with_timeout(self, enhanced_query: str, timeout_seconds: int = 30) -> Dict[str, Any]:
        """
        Executa a consulta com timeout para garantir performance.
        
        Args:
            enhanced_query: Consulta enriquecida com contexto de tabelas relevantes
            timeout_seconds: Tempo máximo em segundos para execução
            
        Returns:
            Resultados da consulta com metadados
        """
        # Usar o handler para capturar a consulta SQL (usar nosso callback existente)
        self.sql_query = None  # Resetar consulta capturada
        self.start_time = None  # Resetar tempo de início
        self.exec_time = None  # Resetar tempo de execução
        
        try:
            import concurrent.futures
            import threading
            
            # Variável para armazenar o resultado
            result_container = {"result": None, "done": False}
            
            # Função que executa a consulta
            def execute_query():
                try:
                    result = self.agent.run(enhanced_query, callbacks=[self])
                    result_container["result"] = {
                        "success": True,
                        "result": result,
                        "sql_query": self.get_captured_query(),
                        "execution_time": self.get_execution_time()
                    }
                except Exception as e:
                    result_container["result"] = {
                        "success": False,
                        "error": str(e)
                    }
                finally:
                    result_container["done"] = True
            
            # Iniciar a thread de execução
            query_thread = threading.Thread(target=execute_query)
            query_thread.daemon = True
            query_thread.start()
            
            # Aguardar com timeout
            start_wait = time.time()
            while not result_container["done"] and time.time() - start_wait < timeout_seconds:
                time.sleep(0.1)
            
            # Verificar resultado ou timeout
            if not result_container["done"]:
                return {
                    "success": False,
                    "error": f"Timeout: A consulta excedeu o limite de {timeout_seconds} segundos"
                }
            
            return result_container["result"]
                    
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _extract_tables_from_query(self, sql_query: str) -> List[str]:
        """
        Extrai nomes de tabelas utilizadas na consulta SQL.
        
        Args:
            sql_query: Consulta SQL
            
        Returns:
            Lista de nomes de tabelas utilizadas
        """
        if not sql_query:
            return []
        
        import re
        
        # Normalizar consulta
        sql_normalized = " " + sql_query.lower() + " "
        
        # Padrão para detectar tabelas após FROM ou JOIN
        table_pattern = r'(?:from|join)\s+([a-z0-9_]+)'
        
        # Encontrar todas as correspondências
        tables = []
        for match in re.finditer(table_pattern, sql_normalized):
            table = match.group(1).strip()
            if table and table not in ['select', 'where', 'group', 'order', 'having', 'limit']:
                tables.append(table)
        
        return list(set(tables))  # Remover duplicatas
    
    def _generate_cache_key(self, user_question: str) -> str:
        """
        Gera uma chave de cache para a consulta do usuário.
        
        Args:
            user_question: Pergunta do usuário
            
        Returns:
            Chave de hash para o cache
        """
        normalized_question = user_question.lower().strip()
        return hashlib.md5(normalized_question.encode()).hexdigest()
    
    def get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Recupera resultado do cache se disponível e válido.
        
        Args:
            key: Chave do cache (hash da consulta)
            
        Returns:
            Valor armazenado ou None se não encontrado/expirado
        """
        if not hasattr(self, 'query_cache'):
            self.query_cache = {}
            return None
        
        if key not in self.query_cache:
            return None
        
        # Verificar TTL
        cache_entry = self.query_cache[key]
        current_time = time.time()
        timestamp = cache_entry.get("timestamp", 0)
        ttl = cache_entry.get("ttl", 3600)
        
        # Verificar expiração
        if current_time - timestamp > ttl:
            # Cache expirado, remover entrada
            del self.query_cache[key]
            print(f"Cache expirado: {key[:8]}...")
            return None
        
        # Cache válido
        print(f"Cache hit: {key[:8]}... (Idade: {int(current_time - timestamp)}s)")
        return cache_entry.get("value")
    
    def add_to_cache(self, key: str, value: Dict[str, Any], ttl: int = 3600) -> None:
        """
        Adiciona resultado ao cache com TTL específico.
        
        Args:
            key: Chave do cache (hash da consulta)
            value: Valor a ser armazenado
            ttl: Tempo de vida em segundos (padrão: 1 hora)
        """
        if not hasattr(self, 'query_cache'):
            self.query_cache = {}
        
        # Adicionar timestamp para controle de TTL
        cache_entry = {
            "timestamp": time.time(),
            "ttl": ttl,
            "value": value
        }
        
        # Armazenar em cache
        self.query_cache[key] = cache_entry
        
        # Log para monitoramento
        print(f"Adicionado ao cache: {key[:8]}... (TTL: {ttl}s)")
        
    def _is_large_database(self) -> bool:
        """
        Determina se estamos lidando com um banco de dados grande.
        
        Critérios para considerar um banco "grande":
        1. Mais de 200 tabelas
        2. Múltiplos módulos Odoo detectados
        
        Returns:
            True se o banco for considerado grande, False caso contrário
        """
        # Se já temos a contagem de tabelas, usamos ela
        if hasattr(self, '_table_count'):
            return self._table_count > 200
        
        # Caso contrário, contamos as tabelas
        try:
            tables = self.schema_extractor.get_all_tables()
            self._table_count = len(tables)
            
            # Verificar contagem de módulos Odoo também
            odoo_modules = set()
            for table in tables:
                module = self.schema_extractor._infer_odoo_module(table)
                if module:
                    odoo_modules.add(module)
            
            self._odoo_module_count = len(odoo_modules)
            
            # Consideramos grande se tiver muitas tabelas OU muitos módulos
            return self._table_count > 200 or self._odoo_module_count > 8
            
        except Exception as e:
            print(f"Erro ao determinar tamanho do banco: {str(e)}")
            # Em caso de erro, assumimos banco normal
            return False
    
    def check_and_adapt_query_from_examples(self, query: str) -> str:
        """Verifica se a consulta deve ser substituída por um exemplo adaptado.
        
        Esta função analisa a pergunta do usuário e a compara com exemplos conhecidos,
        buscando correspondências que permitam adaptar a consulta corretamente. Também
        aplica adaptações genéricas para casos que não correspondem a exemplos específicos.
        
        Args:
            query: A consulta SQL original a ser potencialmente substituída
            
        Returns:
            str: A consulta adaptada de um exemplo ou a consulta original com adaptações genéricas
        """
        if not self.user_question or self.example_similarity_computed:
            return query
        
        # Marca que já computamos a similaridade com exemplos para evitar processamento repetido
        self.example_similarity_computed = True
        
        # Prepara a pergunta do usuário para comparação
        def normalize_text(text):
            return ' '.join(text.lower().replace('?', '').replace(',', '').replace('.', '').split())
        
        normalized_user_question = normalize_text(self.user_question)
        
        # Processamento dos exemplos
        from difflib import SequenceMatcher
        import re
        
        # Extrair os pares pergunta-consulta dos exemplos
        from modules.examples import FEW_SHOT_EXAMPLES
        example_pairs = []
        current_section = None
        
        for line in FEW_SHOT_EXAMPLES.split('\n'):
            if line.lower().startswith('pergunta:'):
                current_section = {'question': line[len('pergunta:'):].strip(), 'sql': ''}
                example_pairs.append(current_section)
            elif current_section and line.startswith('SQL:'):
                continue
            elif current_section and '```sql' in line:
                current_section['sql_started'] = True
                current_section['sql'] = ''
            elif current_section and current_section.get('sql_started') and '```' in line and not '```sql' in line:
                current_section['sql_started'] = False
            elif current_section and current_section.get('sql_started'):
                current_section['sql'] += line + '\n'
        
        # Padrões para detecção
        nivel_estoque_keywords = ['nivel', 'estoque', 'produtos', 'vendidos', 'valor']
        produtos_sem_estoque_keywords = ['produtos', 'vendidos', 'dias', 'estoque', 'tem', 'maos']
        
        ano_pattern = re.compile(r'\b(19|20)\d{2}\b')  # Detecta anos entre 1900 e 2099
        dias_pattern = re.compile(r'\b(\d+)\s*(dias|dias)\b')  # Detecta padrões como "30 dias", "60 dias"
        
        user_has_ano_pattern = bool(ano_pattern.search(normalized_user_question))
        user_has_dias_pattern = bool(dias_pattern.search(normalized_user_question))
        
        # 1. Verificar correspondência com exemplo de nível de estoque
        if all(kw in normalized_user_question for kw in ['nivel', 'estoque', 'produtos', 'vendidos']) and 'valor' in normalized_user_question and user_has_ano_pattern:
            for example in example_pairs:
                if example.get('sql'):
                    normalized_example_question = normalize_text(example['question'])
                    # Verificar se é o exemplo de nível de estoque
                    if all(kw in normalized_example_question for kw in nivel_estoque_keywords) and 'valor' in normalized_example_question:
                        print(f"Encontrado exemplo correspondente para nível de estoque dos produtos mais vendidos")
                        
                        # Extrair o ano da pergunta do usuário
                        ano_match = ano_pattern.search(normalized_user_question)
                        if ano_match:
                            ano = ano_match.group(0)
                            sql = example['sql'].strip()
                            
                            # Procurar o padrão de ano no SQL e substituí-lo
                            ano_sql_pattern = re.compile(r'AND\s+EXTRACT\s*\(\s*YEAR\s+FROM\s+[^\)]+\)\s*=\s*(\d{4})')
                            ano_sql_match = ano_sql_pattern.search(sql)
                            
                            if ano_sql_match:
                                ano_original = ano_sql_match.group(1)
                                if ano != ano_original:
                                    print(f"Adaptando consulta para o ano {ano} em vez de {ano_original}")
                                    sql = sql.replace(f"= {ano_original}", f"= {ano}")
                                    
                                    # Extrai o número da LIMIT se presente na pergunta
                                    num_pattern = re.compile(r'\b(\d+)\s*produtos')
                                    num_match = num_pattern.search(normalized_user_question)
                                    if num_match:
                                        num_produtos = num_match.group(1)
                                        # Substituir o LIMIT original pelo solicitado
                                        limit_pattern = re.compile(r'LIMIT\s+(\d+)')
                                        limit_match = limit_pattern.search(sql)
                                        if limit_match:
                                            limit_original = limit_match.group(1)
                                            sql = sql.replace(f"LIMIT {limit_original}", f"LIMIT {num_produtos}")
                                    
                                    print("Usando consulta adaptada do exemplo de nível de estoque")
                                    self.usando_exemplo_adaptado = True
                                    return sql
        
        # 2. Verificar correspondência com exemplo de produtos vendidos sem estoque
        if ('produtos' in normalized_user_question or 'quais' in normalized_user_question) and \
           'vendidos' in normalized_user_question and \
           ('ultimos' in normalized_user_question or 'recentes' in normalized_user_question or user_has_dias_pattern) and \
           ('estoque' in normalized_user_question and ('nao' in normalized_user_question or 'sem' in normalized_user_question or 'tem' in normalized_user_question or 'maos' in normalized_user_question)):
            
            for example in example_pairs:
                if example.get('sql'):
                    normalized_example_question = normalize_text(example['question'])
                    
                    # Verificar se é o exemplo de produtos sem estoque
                    produtos_sem_estoque_score = sum(1 for kw in produtos_sem_estoque_keywords if kw in normalized_example_question)
                    if produtos_sem_estoque_score >= 4 and 'ultimos' in normalized_example_question and 'dias' in normalized_example_question:
                        print(f"Encontrado exemplo correspondente para produtos vendidos sem estoque")
                        
                        sql = example['sql'].strip()
                        
                        # Extrair o número de dias da pergunta do usuário
                        dias_match = dias_pattern.search(normalized_user_question)
                        if dias_match:
                            # Extrair o número de dias
                            num_dias = dias_match.group(1)
                            
                            # Procurar o padrão de dias no SQL
                            dias_sql_pattern = re.compile(r"INTERVAL\s+'(\d+)\s+days'")
                            dias_sql_match = dias_sql_pattern.search(sql)
                            
                            if dias_sql_match:
                                dias_original = dias_sql_match.group(1)
                                if num_dias != dias_original:
                                    print(f"Adaptando consulta para {num_dias} dias em vez de {dias_original} dias")
                                    sql = sql.replace(f"'{dias_original} days'", f"'{num_dias} days'")
                        
                            print("Usando consulta adaptada do exemplo de produtos sem estoque")
                            self.usando_exemplo_adaptado = True
                            return sql
        
        # Se chegou aqui, nenhum exemplo correspondente foi encontrado
        # Tenta aplicar adaptações genéricas
        return self.apply_generic_adaptations(query, self.user_question)

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
            input_data = None
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

            # Armazena e potencialmente modifica a query se for uma SELECT
            if potential_query and isinstance(potential_query, str) and "SELECT" in potential_query.upper():
                # Verifica se devemos adaptar a query baseado em exemplos
                adapted_query = self.check_and_adapt_query_from_examples(potential_query)
                
                # Se a query foi adaptada, substitua no input original
                if self.usando_exemplo_adaptado and adapted_query != potential_query:
                    if input_data and isinstance(input_data, dict):
                        input_data['query'] = adapted_query
                        # Substitui o input original pela versão modificada
                        serialized["inputs"] = json.dumps(input_data)
                    else:
                        serialized["inputs"] = adapted_query
                    
                    print(f"--- Consulta adaptada para exemplo: {adapted_query} ---")
                
                # Armazena a query final (original ou adaptada)
                self.sql_query = adapted_query if self.usando_exemplo_adaptado else potential_query
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


class QuerySQLCheckerTool:
    """Ferramenta para verificar e validar consultas SQL antes da execução.

    Esta ferramenta usa um LLM para verificar se uma consulta SQL contém erros comuns
    ou comandos potencialmente perigosos antes de executá-la no banco de dados.

    Adaptado de: https://www.patterns.app/blog/2023/1/18/crunchbot-sql-analyst-gpt/
    """

    def __init__(self, llm: ChatOpenAI, db: SQLDatabase):
        """Inicializa o validador de consultas SQL.

        Args:
            llm: Modelo de linguagem para verificar as consultas
            db: Banco de dados SQL para obter informações sobre o dialeto
        """
        self.llm = llm
        self.db = db

        # Cria o prompt para validação de consultas SQL
        self.prompt = PromptTemplate(
            template=QUERY_CHECKER,
            input_variables=["dialect", "query"]
        )

        # Cria a sequência de execução usando a nova abordagem recomendada
        # prompt | llm em vez de LLMChain
        self.chain = self.prompt | self.llm

    def validate_query(self, query: str) -> Tuple[bool, str]:
        """Valida uma consulta SQL usando o LLM.

        Args:
            query: A consulta SQL a ser validada

        Returns:
            Tupla contendo (is_valid, result_or_error)
            - is_valid: True se a consulta for válida, False caso contrário
            - result_or_error: A consulta corrigida ou uma mensagem de erro
        """
        # Verifica se a consulta é um exemplo do arquivo examples.py
        from .examples import FEW_SHOT_EXAMPLES
        
        # Verifica se a consulta está nos exemplos
        is_example_query = False
        normalized_query = ' '.join(query.strip().split())
        
        for example in FEW_SHOT_EXAMPLES.split("SQL:"):
            if len(example.strip()) > 0 and "```sql" in example:
                # Extrai a consulta SQL do exemplo
                sql_part = example.split("```sql")[1].split("```")[0].strip()
                normalized_example = ' '.join(sql_part.strip().split())
                
                # Compara os textos normalizados para evitar problemas de espaçamento
                if normalized_query == normalized_example:
                    print("Consulta identificada como exemplo predefinido. Pulando validação.")
                    return True, query
                
                # Verifica se a consulta está contida no exemplo ou vice-versa
                # Isso ajuda a identificar casos onde o LLM pegou apenas parte do exemplo
                if len(normalized_query) > 50 and len(normalized_example) > 50:
                    if normalized_query in normalized_example or normalized_example in normalized_query:
                        print("Consulta identificada como parte de um exemplo predefinido. Pulando validação.")
                        return True, query
        
        try:
            # Verifica se a consulta contém comandos não permitidos
            dangerous_commands = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE"]
            for cmd in dangerous_commands:
                if cmd in query.upper():
                    return False, f"Erro: Consulta contém comando não permitido: {cmd}. Apenas consultas SELECT são permitidas."

            # Verifica se a consulta é complexa (contém CTEs ou subconsultas)
            is_complex_query = "WITH" in query.upper() or query.upper().count("SELECT") > 1

            # Para consultas complexas, fazemos uma validação mais simples para evitar truncamento
            if is_complex_query:
                print("Detectada consulta complexa com CTEs ou subconsultas")

                # Verifica apenas se a consulta começa com SELECT ou WITH
                if query.upper().strip().startswith("SELECT") or query.upper().strip().startswith("WITH"):
                    # Verifica se a consulta parece completa (tem ponto e vírgula no final ou não termina abruptamente)
                    if query.strip().endswith(";") or query.strip().endswith(")"):
                        print("Consulta complexa validada sem modificações")
                        return True, query

                # Se a consulta não parece estar completa, tenta validar normalmente
                print("Consulta complexa pode estar incompleta, tentando validação completa")

            # Usa o LLM para verificar e corrigir a consulta
            result = self.chain.invoke({
                "query": query,
                "dialect": self.db.dialect
            }).content

            # Verifica se o resultado é muito curto comparado com a consulta original
            # Isso pode indicar que o LLM truncou a resposta
            if len(result) < len(query) * 0.5 and is_complex_query:
                print("Possível truncamento detectado na validação. Mantendo consulta original.")
                return True, query

            # Se o resultado for diferente da consulta original, considera que houve correção
            if result.strip() != query.strip():
                print(f"--- Consulta corrigida pelo validador ---")
                print(f"Original: {query}")
                print(f"Corrigida: {result}")

                # Verifica se a correção manteve a estrutura básica da consulta
                if is_complex_query and "WITH" in query.upper() and "WITH" not in result.upper():
                    print("A correção removeu a estrutura CTE. Mantendo consulta original.")
                    return True, query

                return True, result

            # Se não houve alteração, a consulta é válida
            return True, query

        except Exception as e:
            print(f"Erro na validação: {str(e)}")
            # Em caso de erro na validação, preferimos manter a consulta original
            # para consultas complexas, em vez de falhar completamente
            if "WITH" in query.upper() or query.upper().count("SELECT") > 1:
                print("Erro ao validar consulta complexa. Mantendo consulta original.")
                return True, query
            return False, f"Erro ao validar consulta: {str(e)}"


class OdooTextToSQL:
    """Classe principal otimizada para processamento de consultas text-to-SQL em bancos Odoo

    Esta classe implementa um agente SQL que converte perguntas em linguagem natural para consultas SQL.
    Utiliza técnicas avançadas como:
    - Embeddings para identificar tabelas relevantes
    - Exemplos few-shot para melhorar a precisão das consultas geradas
    - Cache de resultados para otimizar performance
    - Callbacks para capturar as consultas SQL geradas

    Os exemplos few-shot são utilizados para ensinar ao modelo como gerar consultas SQL para perguntas
    comuns em um banco de dados Odoo ERP. Estes exemplos são combinados com o prefixo de segurança
    para criar um prompt mais eficaz para o agente SQL.
    """

    def __init__(self, db_uri: str, use_checkpoint: bool = True, force_reprocess: bool = False,
                data_dir: Optional[str] = None, enable_llm_cache: bool = True):
        """Inicializa o agente SQL com configurações personalizadas

        Args:
            db_uri: URI de conexão com o banco de dados
            use_checkpoint: Se deve usar checkpoints salvos para embeddings
            force_reprocess: Se deve forçar o reprocessamento dos embeddings
            data_dir: Diretório para armazenar dados e checkpoints
            enable_llm_cache: Se deve habilitar cache para chamadas ao LLM
        """
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

        # Combina o prefixo de segurança com os exemplos few-shot
        combined_prefix = f"{SAFETY_PREFIX}\n\n{FEW_SHOT_EXAMPLES}"

        # --- Usa o prefixo combinado com exemplos few-shot ---
        self.agent = create_sql_agent(
            llm=self.llm,
            toolkit=self.toolkit,
            verbose=True,
            agent_type="openai-tools",
            prefix=combined_prefix  # Usa a combinação de prefixo de segurança e exemplos
        )
        # Instancia o callback handler
        self.query_callback_handler = SQLQueryCaptureCallback()
        # Cache para resultados de consultas
        self.query_cache = {}
        # Inicializa o validador de consultas SQL
        self.query_checker = QuerySQLCheckerTool(llm=self.llm, db=self.db)

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
        """Encontra tabelas relevantes para a consulta com abordagem hierárquica

        Utiliza uma estratégia de busca em múltiplas camadas para identificar as tabelas
        mais relevantes para a consulta do usuário:
        1. Busca documentos de visão geral e módulos para contexto
        2. Busca documentos de estrutura de tabelas (principal fonte)
        3. Busca documentos de dados de tabelas (complemento)
        4. Se necessário, faz uma busca genérica para garantir resultados

        Args:
            query: Pergunta do usuário em linguagem natural
            top_k: Número máximo de tabelas a retornar

        Returns:
            Lista de nomes de tabelas relevantes para a consulta
        """
        # Primeiro, busca documentos de visão geral e módulos
        # Estes documentos ajudam a entender o contexto, mas não são usados diretamente
        # para extrair nomes de tabelas
        self.vector_store.similarity_search(
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
        # Filtra tabelas importantes usando um método público
        important_tables = []
        for table in tables:
            # Verifica se é uma tabela importante usando a categoria
            category = self.schema_extractor._identify_table_category(table)
            if category in ["main_business", "transactional"]:
                important_tables.append(table)
                if len(important_tables) >= 3:
                    break

        for table in important_tables:
            data_doc = self.schema_extractor.extract_table_data_document(table)
            if data_doc:
                sample_data_info.append(data_doc.page_content)

        # Determina se é um banco Odoo para adicionar contexto específico
        odoo_context = ""
        if self.schema_extractor.is_odoo:
            odoo_context = ODOO_CONTEXT

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

    def get_few_shot_examples(self) -> str:
        """Retorna os exemplos few-shot utilizados pelo agente SQL

        Os exemplos few-shot são utilizados para melhorar a precisão das consultas SQL geradas.
        Cada exemplo consiste em uma pergunta em linguagem natural e a consulta SQL correspondente.
        Estes exemplos ajudam o modelo a entender o padrão de consultas esperado para o banco Odoo.

        Returns:
            String contendo os exemplos few-shot formatados
        """
        return FEW_SHOT_EXAMPLES

    def query(self, user_question: str) -> Tuple[Dict[str, Any], Optional[str]]:
        """Processa pergunta do usuário e gera consulta SQL

        Este método implementa o fluxo completo de processamento:
        1. Verifica cache para resultados existentes
        2. Identifica tabelas relevantes para a consulta
        3. Enriquece a pergunta com informações de schema
        4. Executa o agente SQL com exemplos few-shot
        5. Captura a consulta SQL gerada via callback
        6. Armazena o resultado em cache para uso futuro

        O uso de exemplos few-shot melhora significativamente a qualidade das consultas geradas,
        especialmente para perguntas complexas que envolvem múltiplas tabelas e joins.

        Args:
            user_question: Pergunta do usuário em linguagem natural

        Returns:
            Tupla contendo o resultado do agente e a consulta SQL capturada
        """
        # Verifica primeiro o cache
        cached_data = self.get_query_from_cache(user_question)
        if cached_data:
            print("Usando resultado em cache")
            return cached_data['result'], cached_data['query']

        # Reseta o callback antes de cada consulta
        self.query_callback_handler.reset()
        # Passa a pergunta do usuário para o callback para possível análise de exemplos
        self.query_callback_handler.set_user_question(user_question)

        # Encontra tabelas relevantes para a consulta
        relevant_tables = self.find_relevant_tables(user_question)
        print(f"Tabelas relevantes: {', '.join(relevant_tables)}")

        # Enriquece a pergunta com informações de schema
        enhanced_question = self.enhance_query_with_table_info(user_question, relevant_tables)

        # Registra tempo de início
        start_time = time.time()

        # Executa o agente passando o callback handler
        # O agente utiliza os exemplos few-shot definidos na inicialização
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

        # Primeiro, verifica se a consulta corresponde a algum dos exemplos few-shot
        # Se corresponder, usamos a consulta do exemplo diretamente sem validação
        is_from_few_shot = False
        exact_match_found = False
        query_from_example = None
        
        # Prepara a pergunta do usuário para comparação
        def normalize_text(text):
            return ' '.join(text.lower().replace('?', '').replace(',', '').replace('.', '').split())
            
        normalized_user_question = normalize_text(user_question)
        print(f"Procurando exemplo para: '{normalized_user_question}'")
        
        # Processar os exemplos de FEW_SHOT_EXAMPLES
        from difflib import SequenceMatcher
        
        # Armazenar as perguntas e consultas dos exemplos
        example_pairs = []
        
        # Extrair os pares pergunta-consulta dos exemplos
        current_section = None
        for line in FEW_SHOT_EXAMPLES.split('\n'):
            if line.lower().startswith('pergunta:'):
                current_section = {'question': line[len('pergunta:'):].strip(), 'sql': ''}
                example_pairs.append(current_section)
            elif current_section and line.startswith('SQL:'):
                # A seção SQL começa na próxima linha após este marcador
                continue
            elif current_section and '```sql' in line:
                # Começa a capturar o SQL
                current_section['sql_started'] = True
                current_section['sql'] = ''
            elif current_section and current_section.get('sql_started') and '```' in line and not '```sql' in line:
                # Termina a captura do SQL
                current_section['sql_started'] = False
            elif current_section and current_section.get('sql_started'):
                # Adiciona a linha à consulta SQL
                current_section['sql'] += line + '\n'
        
        # Usar similaridade para encontrar o melhor exemplo
        best_match = None
        best_similarity = 0
        produtos_sem_estoque_example = None
        produtos_mais_vendidos_example = None
        
        for example in example_pairs:
            if example.get('sql'):
                # Normalizar pergunta do exemplo
                normalized_example_question = normalize_text(example['question'])
                
                # Calcular similaridade
                similarity = SequenceMatcher(None, normalized_user_question, normalized_example_question).ratio()
                
                # Verificar se é o exemplo de produtos vendidos sem estoque
                produtos_sem_estoque_keywords = ['produtos', 'vendidos', 'dias', 'estoque', 'maos']
                produtos_sem_estoque_score = sum(1 for kw in produtos_sem_estoque_keywords if kw in normalized_example_question)
                is_produtos_sem_estoque = produtos_sem_estoque_score >= 3 and 'tem estoque' in normalized_example_question
                
                # Verificar se é o exemplo de nível de estoque dos mais vendidos
                nivel_estoque_keywords = ['nivel', 'estoque', 'produtos', 'vendidos', 'valor']
                nivel_estoque_score = sum(1 for kw in nivel_estoque_keywords if kw in normalized_example_question)
                is_nivel_estoque = nivel_estoque_score >= 3 and 'valor' in normalized_example_question
                
                # Salvar os exemplos específicos para uso posterior
                if is_produtos_sem_estoque:
                    produtos_sem_estoque_example = example
                if is_nivel_estoque:
                    produtos_mais_vendidos_example = example
                
                # Verificar se é o melhor match
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = example
                    
                # Uso de expressões regulares para detectar padrões numéricos
                import re
                dias_pattern = re.compile(r'\d+\s*dias')
                ano_pattern = re.compile(r'\b(19|20)\d{2}\b')  # Detecta anos entre 1900 e 2099
                user_has_dias_pattern = bool(dias_pattern.search(normalized_user_question))
                user_has_ano_pattern = bool(ano_pattern.search(normalized_user_question))
                
                # Verificar perguntas sobre produtos vendidos sem estoque com variações numéricas
                if is_produtos_sem_estoque and 'produtos' in normalized_user_question and 'vendidos' in normalized_user_question and \
                   (user_has_dias_pattern or 'ultimo' in normalized_user_question or 'recente' in normalized_user_question) and \
                   ('estoque' in normalized_user_question or 'em maos' in normalized_user_question):
                    print(f"Encontrada correspondência para pergunta sobre produtos vendidos sem estoque")
                    
                    # Extrair o número de dias da pergunta do usuário, se existir
                    dias_match = dias_pattern.search(normalized_user_question)
                    if dias_match:
                        # Extrair o número de dias da pergunta do usuário
                        dias_str = dias_match.group(0)
                        numero_dias = ''.join(filter(str.isdigit, dias_str))
                        
                        if numero_dias and numero_dias != '30':
                            print(f"Adaptando consulta para {numero_dias} dias em vez de 30 dias")
                            # Adaptar a consulta SQL para o número de dias especificado
                            query_from_example = example['sql'].strip().replace("'30 days'", f"'{numero_dias} days'")
                        else:
                            query_from_example = example['sql'].strip()
                    else:
                        query_from_example = example['sql'].strip()
                        
                    exact_match_found = True
                    break
                
                # Verificar perguntas sobre nível de estoque dos produtos mais vendidos em valor
                elif is_nivel_estoque and 'nivel' in normalized_user_question and 'estoque' in normalized_user_question and \
                     'produtos' in normalized_user_question and 'vendidos' in normalized_user_question and \
                     'valor' in normalized_user_question and user_has_ano_pattern:
                    print(f"Encontrada correspondência para pergunta sobre nível de estoque dos produtos mais vendidos em valor")
                    
                    # Extrair o ano da pergunta do usuário
                    ano_match = ano_pattern.search(normalized_user_question)
                    if ano_match:
                        # Extrair o ano da pergunta do usuário
                        ano = ano_match.group(0)
                        
                        # Pegar o SQL do exemplo e adaptar o ano
                        query_from_example = example['sql'].strip()
                        
                        # Procurar o padrão de ano no SQL e substituí-lo
                        ano_sql_pattern = re.compile(r'AND\s+EXTRACT\s*\(\s*YEAR\s+FROM\s+[^\)]+\)\s*=\s*(\d{4})')
                        ano_sql_match = ano_sql_pattern.search(query_from_example)
                        
                        if ano_sql_match:
                            ano_original = ano_sql_match.group(1)
                            if ano != ano_original:
                                print(f"Adaptando consulta para o ano {ano} em vez de {ano_original}")
                                query_from_example = query_from_example.replace(f"= {ano_original}", f"= {ano}")
                    else:
                        query_from_example = example['sql'].strip()
                        
                    exact_match_found = True
                    break
                    
        # Se encontramos uma correspondência exata
        if exact_match_found and query_from_example:
            print("Usando consulta do exemplo específico (correspondência exata)")
            captured_query = query_from_example
            is_from_few_shot = True
        # Se temos o exemplo de produtos sem estoque e a pergunta parece ser sobre isso
        elif produtos_sem_estoque_example and 'produtos' in normalized_user_question and 'vendidos' in normalized_user_question and \
             ('estoque' in normalized_user_question or 'em maos' in normalized_user_question):
            print("Usando consulta do exemplo de produtos vendidos sem estoque (correspondência por palavras-chave)")
            captured_query = produtos_sem_estoque_example['sql'].strip()
            is_from_few_shot = True
        # Se temos o exemplo de nível de estoque e a pergunta parece ser sobre isso
        elif produtos_mais_vendidos_example and 'nivel' in normalized_user_question and 'estoque' in normalized_user_question and \
             'produtos' in normalized_user_question and 'valor' in normalized_user_question:
            print("Usando consulta do exemplo de nível de estoque dos produtos mais vendidos (correspondência por palavras-chave)")
            
            # Verificar se há um ano mencionado
            ano_match = ano_pattern.search(normalized_user_question) if 'ano_pattern' in locals() else None
            if ano_match:
                ano = ano_match.group(0)
                sql = produtos_mais_vendidos_example['sql'].strip()
                
                # Procurar o padrão de ano no SQL e substituí-lo
                ano_sql_pattern = re.compile(r'AND\s+EXTRACT\s*\(\s*YEAR\s+FROM\s+[^\)]+\)\s*=\s*(\d{4})') if 're' in locals() else None
                if ano_sql_pattern:
                    ano_sql_match = ano_sql_pattern.search(sql)
                    if ano_sql_match:
                        ano_original = ano_sql_match.group(1)
                        if ano != ano_original:
                            print(f"Adaptando consulta para o ano {ano} em vez de {ano_original}")
                            sql = sql.replace(f"= {ano_original}", f"= {ano}")
                
                captured_query = sql
            else:
                captured_query = produtos_mais_vendidos_example['sql'].strip()
            
            is_from_few_shot = True
        # Se temos um match com alta similaridade
        elif best_match and best_similarity > 0.7:
            print(f"Usando consulta do exemplo mais similar (similaridade: {best_similarity:.2f})")
            captured_query = best_match['sql'].strip()
            is_from_few_shot = True
        
        # Se a verificação automática falhou, podemos continuar com as verificações originais
        if not is_from_few_shot:
            print("Usando método tradicional de verificação de exemplos")
            for example in FEW_SHOT_EXAMPLES.split("SQL:"):
                if len(example.strip()) > 0 and "```sql" in example:
                    # Extrai a consulta SQL do exemplo
                    sql_part = example.split("```sql")[1].split("```")[0].strip()
                    
                    # Normaliza a consulta para comparação (remove espaços extras e quebras de linha)
                    normalized_sql_part = ' '.join(sql_part.replace('\n', ' ').split())
                    normalized_captured_query = '' if not captured_query else ' '.join(captured_query.replace('\n', ' ').split())

                    # Verifica se a pergunta do usuário corresponde a este exemplo
                    question_part = example.split("Pergunta:")[1].split("SQL:")[0].strip() if "Pergunta:" in example else ""
                    
                    # Comparação de perguntas mais flexível - verifica se as palavras-chave principais coincidem
                    if question_part and all(word in user_question.lower() for word in question_part.lower().split()[:5]):
                        print(f"Encontrado exemplo few-shot correspondente às palavras-chave da pergunta")
                        captured_query = sql_part
                        is_from_few_shot = True
                        break

                    # Verifica se a consulta capturada é idêntica ou muito similar à consulta do exemplo
                    if captured_query and (normalized_captured_query == normalized_sql_part or \
                                       (len(normalized_captured_query) > 50 and normalized_sql_part.startswith(normalized_captured_query[:50]))):
                        print(f"Consulta capturada corresponde a um exemplo few-shot")
                        captured_query = sql_part
                        is_from_few_shot = True
                        break

                    # Verifica se a consulta capturada é um fragmento da consulta do exemplo
                    if captured_query and (normalized_captured_query in normalized_sql_part or normalized_sql_part in normalized_captured_query):
                        print(f"Consulta capturada parece ser um fragmento de um exemplo few-shot")
                        captured_query = sql_part
                        is_from_few_shot = True
                        break
                        
                    # Verificação por tabelas específicas e estrutura de consulta
                    if captured_query and 'sale_order_line' in captured_query and 'stock_quant' in captured_query and 'COALESCE' in captured_query:
                        print(f"Estrutura da consulta corresponde ao exemplo de produtos vendidos sem estoque")
                        # Busca o exemplo específico
                        if 'produtos foram vendidos nos últimos 30 dias' in example:
                            captured_query = sql_part
                            is_from_few_shot = True
                            break

        # Se não for de um exemplo few-shot, verifica se é uma consulta complexa
        if not is_from_few_shot and captured_query:
            is_complex_query = ("WITH" in captured_query.upper() or captured_query.upper().count("SELECT") > 1)

            if is_complex_query:
                print("Detectada consulta complexa com CTEs ou subconsultas")

                # Verifica se a consulta parece estar completa
                if not (captured_query.strip().endswith(";") or captured_query.strip().endswith(")")):
                    print("Consulta complexa pode estar incompleta, verificando...")

        # Valida a consulta SQL antes de retornar, exceto se for de um exemplo few-shot
        if captured_query:
            if is_from_few_shot:
                print("Consulta é de um exemplo few-shot, pulando validação")
                # Se encontramos um match exato, use query_from_example, caso contrário use captured_query
                if exact_match_found and query_from_example:
                    final_query = query_from_example
                else:
                    final_query = captured_query
            else:
                print("Validando consulta SQL...")
                is_valid, validated_query = self.query_checker.validate_query(captured_query)

                if not is_valid:
                    print(f"ERRO DE VALIDAÇÃO: {validated_query}")
                    # Se a consulta for inválida, adiciona o erro ao resultado
                    if isinstance(result, dict) and "output" in result:
                        result["output"] += f"\n\nERRO DE VALIDAÇÃO: {validated_query}"
                    # Retorna a consulta original para referência
                    final_query = captured_query
                else:
                    print("Consulta validada com sucesso!")
                    # Se a consulta foi corrigida, usa a versão corrigida
                    final_query = validated_query

                    # Se a consulta foi corrigida, atualiza o resultado
        # Adiciona ao cache
        self.add_query_to_cache(user_question, result, final_query)

        # Retorna tanto o resultado quanto a query validada
        return result, final_query
