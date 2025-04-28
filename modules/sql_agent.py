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
import re # Adicionar importação de regex
import logging # Adicionar importação de logging

# --- Novas Importações para Few-Shot Examples ---
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
# ----------------------------------------------

# Configura logging básico
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    
    # --- Dicionário de descrições para tabelas chave ---
    ODOO_TABLE_DESCRIPTIONS = {
        "res_partner": "Contém informações de Contatos, Clientes e Fornecedores. Chave para relacionar vendas, compras e faturas a entidades.",
        "product_template": "Modelo base para Produtos (informações gerais como nome, tipo, categoria).",
        "product_product": "Variantes específicas de Produtos (produto real estocável/vendável). Liga-se a product_template.",
        "sale_order": "Cabeçalho do Pedido de Venda (cotação, venda). Contém cliente (partner_id), data (date_order), status (state), valor total (amount_total). Essencial para análise de vendas.",
        "sale_order_line": "Linhas/Itens de um Pedido de Venda. Contém produto (product_id), quantidade (product_uom_qty), preço unitário (price_unit), subtotal (price_subtotal). Liga-se a sale_order.",
        "purchase_order": "Cabeçalho do Pedido de Compra. Contém fornecedor (partner_id), data, status, valor total.",
        "purchase_order_line": "Linhas/Itens de um Pedido de Compra. Contém produto, quantidade, preço.",
        "account_move": "Lançamentos Contábeis, incluindo Faturas de Cliente (move_type='out_invoice') e Fornecedor (move_type='in_invoice'). Contém parceiro, data, status, valor.",
        "account_move_line": "Linhas de Lançamentos Contábeis/Faturas. Detalha contas, produtos, valores.",
        "stock_quant": "Quantidade atual (estoque em mãos) de um Produto (product_id) em um Local de Estoque (location_id) específico.",
        "stock_move": "Registro de Movimentações de Estoque (transferências) entre locais. Contém produto, quantidade, local de origem e destino.",
        "stock_location": "Locais de Estoque (físicos ou virtuais, como Clientes, Fornecedores).",
        "stock_picking": "Operações de Transferência de Estoque (recebimento, entrega, interno). Agrupa stock_moves."
        # Adicionar mais descrições conforme necessário
    }
    # -------------------------------------------------

    def format_table_info_for_embedding(self, table_info: Dict[str, Any]) -> str:
        """Formata informações da tabela para embedding, incluindo uma descrição aprimorada."""
        table_name = table_info["table_name"]
        category = table_info.get("category", "unknown")
        module = table_info.get("module", "unknown")

        # --- Adicionar Descrição (Lógica Aprimorada) ---
        description = self.ODOO_TABLE_DESCRIPTIONS.get(table_name) # Tenta pegar descrição específica
        if not description: # Se não houver específica, cria uma genérica mais informativa
            description = f"Tabela '{table_name}' (Módulo: {module}, Categoria: {category})."
            if category == "main_business":
                description += " Contém dados principais do negócio (ex: clientes, produtos, vendas, compras)."
            elif category == "transactional":
                description += " Registra transações e operações (ex: linhas de pedido, movimentos contábeis/estoque)."
            elif category == "configuration":
                description += " Armazena configurações e parâmetros do sistema."
            elif category == "technical":
                description += " Tabela técnica interna do sistema Odoo."
            elif category == "log_or_history":
                description += " Registra logs, histórico ou dados para relatórios."
        # ------------------------------------------

        # Formata informações sobre colunas
        column_info = []
        for col in table_info.get("columns", []):
            if not isinstance(col, dict):
                continue
            is_pk = col.get("name") in table_info.get("primary_keys", [])
            pk_str = " (PRIMARY KEY)" if is_pk else ""
            # Garante que o tipo seja string
            col_type_str = str(col.get('type', 'Unknown'))
            col_str = f"- {col.get('name', 'Unknown')}: {col_type_str}{pk_str}"
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
        
        # Informações sobre índices
        index_info = []
        for idx in table_info.get("indexes", []):
            if not isinstance(idx, dict):
                continue
            
            column_names = idx.get("column_names", [])
            if column_names is None:
                column_names = []
            elif isinstance(column_names, str):
                column_names = [column_names]
            
            valid_columns = [str(col) for col in column_names if col is not None]
            
            if valid_columns:
                idx_cols = ", ".join(valid_columns)
                idx_type = "UNIQUE " if idx.get("unique", False) else ""
                idx_str = f"- {idx_type}Index on ({idx_cols})"
                index_info.append(idx_str)
        
        # Amostra de dados
        sample_data_str = ""
        sample_data = table_info.get("sample_data", [])
        if sample_data:
            sample_data_str = "Exemplos de dados:\n"
            for i, row in enumerate(sample_data):
                if isinstance(row, dict):
                    # Converte o dict para string de forma segura
                    try:
                        row_str = json.dumps(row, default=self._json_serializable_converter, ensure_ascii=False)
                        sample_data_str += f"Exemplo {i+1}: {row_str}\n"
                    except Exception:
                        sample_data_str += f"Exemplo {i+1}: (Erro ao serializar linha)\n"
        
        # Monta o documento final com a DESCRIÇÃO adicionada
        formatted_text = f"""
TABLE_NAME: {table_name}
CATEGORY: {category}
MODULE: {module}
DESCRIPTION: {description}

COLUMNS:
{chr(10).join(column_info) if column_info else '-- Nenhuma coluna encontrada'}

FOREIGN KEYS:
{chr(10).join(fk_info) if fk_info else '-- Nenhuma chave estrangeira'}

INDEXES:
{chr(10).join(index_info) if index_info else '-- Nenhum índice definido'}

{sample_data_str}
"""
        # Log do conteúdo formatado para depuração (removido ou manter comentado)
        # logging.debug(f"Conteúdo formatado para embedding da tabela {table_name}:\n{formatted_text[:500]}...")
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
                data_dir: Optional[str] = None, enable_llm_cache: bool = True, num_examples: int = 1):
        config = Config()
        self.db_uri = db_uri
        self.use_checkpoint = use_checkpoint
        self.force_reprocess = force_reprocess
        self.num_examples = num_examples

        # Configuração do diretório de dados
        self.data_dir = data_dir or os.environ.get('SQL_AGENT_DATA_DIR', 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        self.examples_file = os.path.join(self.data_dir, 'sql_examples.json')
        # --- Novo: Diretório para persistir embeddings dos exemplos ---
        self.examples_persist_dir = os.path.join(self.data_dir, 'example_embeddings')
        os.makedirs(self.examples_persist_dir, exist_ok=True) # Garante que o diretório exista
        # ---------------------------------------------------------

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

        # --- Carregar e Configurar Exemplos Few-Shot (modificado) ---
        self.example_selector = self._setup_example_selector()
        # --------------------------------------------

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

    def _load_examples(self) -> List[Dict[str, str]]:
        """Carrega exemplos do arquivo JSON."""
        try:
            if os.path.exists(self.examples_file):
                with open(self.examples_file, 'r', encoding='utf-8') as f:
                    examples = json.load(f)
                    # Valida se é uma lista e se cada item tem 'question' e 'query'
                    if isinstance(examples, list) and all(
                        isinstance(ex, dict) and 'question' in ex and 'query' in ex
                        for ex in examples
                    ):
                        print(f"Carregados {len(examples)} exemplos de {self.examples_file}")
                        return examples
                    else:
                        print(f"AVISO: Formato inválido no arquivo de exemplos: {self.examples_file}. Chaves esperadas 'question' e 'query' não encontradas em todos os itens. Ignorando exemplos.")
                        return []
            else:
                print(f"AVISO: Arquivo de exemplos não encontrado: {self.examples_file}. Nenhum exemplo será carregado.")
                return []
        except json.JSONDecodeError:
            print(f"ERRO: Falha ao decodificar JSON do arquivo de exemplos: {self.examples_file}. Ignorando exemplos.")
            return []
        except Exception as e:
            print(f"ERRO inesperado ao carregar exemplos: {e}. Ignorando exemplos.")
            return []

    def _setup_example_selector(self) -> Optional[SemanticSimilarityExampleSelector]:
        """Configura o seletor de exemplos semânticos usando um Chroma persistido."""
        examples = self._load_examples()
        if not examples:
            return None

        # Prepara os textos e metadados dos exemplos
        texts = [ex.get("question", "") for ex in examples] # Usar .get para segurança
        # Certifica que as chaves corretas estão sendo usadas ('question', 'query')
        metadatas = [{"question": ex.get("question"), "query": ex.get("query")} for ex in examples if ex.get("question") and ex.get("query")]

        if not metadatas:
            print("AVISO: Nenhum exemplo válido encontrado após validação de chaves ('question', 'query').")
            return None

        texts_for_embedding = [m["question"] for m in metadatas] # Apenas as perguntas para embedding

        try:
            # Verifica se o diretório persistido já existe e tem dados
            # (Uma heurística simples: verificar se um arquivo sqlite existe)
            db_file_path = os.path.join(self.examples_persist_dir, "chroma.sqlite3")
            force_recreate_examples = not os.path.exists(db_file_path) # Força recriação se o DB não existe

            if force_recreate_examples:
                print(f"Criando novo vector store de exemplos em: {self.examples_persist_dir}")
                # Cria uma nova instância e adiciona os textos
                example_vector_store = Chroma.from_texts(
                    texts=texts_for_embedding,
                    embedding=self.embeddings,
                    metadatas=metadatas, # Passa os metadados completos
                    persist_directory=self.examples_persist_dir,
                    # collection_name="sql_examples_collection" # Opcional: dar um nome à coleção
                )
                # Nota: from_texts com persist_directory já salva automaticamente
                # example_vector_store.persist() # Não é estritamente necessário aqui
            else:
                print(f"Carregando vector store de exemplos existente de: {self.examples_persist_dir}")
                # Carrega a instância do diretório persistido
                example_vector_store = Chroma(
                    persist_directory=self.examples_persist_dir,
                    embedding_function=self.embeddings
                    # collection_name="sql_examples_collection" # Use se definido na criação
                )
                # TODO: Adicionar lógica para atualizar/reconstruir se sql_examples.json mudar?
                # Por enquanto, ele apenas carrega o que existe.

            example_selector = SemanticSimilarityExampleSelector(
                vectorstore=example_vector_store,
                k=self.num_examples,
                # Certifique-se de que a chave no metadata que contém a pergunta é 'question'
                # A chave de entrada para a *query do usuário* continua sendo 'input' (ou 'query' se preferir padronizar)
                input_keys=["query"] # A chave da query do usuário no dicionário de entrada
            )
            print(f"Seletor de exemplos configurado com k={self.num_examples} usando diretório persistido.")
            return example_selector
        except Exception as e:
            print(f"ERRO CRÍTICO ao configurar o seletor de exemplos com persistência: {e}. Exemplos não serão usados.")
            # Imprime traceback para mais detalhes
            import traceback
            traceback.print_exc()
            return None

    def _get_or_create_schema_embeddings(self) -> Chroma:
        """Cria ou recupera embeddings do schema do banco com estratégia otimizada"""
        try:
            schema_hash = self._generate_schema_hash()
            checkpoint_dir = os.path.join(self.data_dir, 'checkpoints', f"schema_{schema_hash}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            embedding_function = self.embeddings

            # Adiciona verificação explícita para forçar reprocessamento
            if self.force_reprocess and os.path.exists(os.path.join(checkpoint_dir, "chroma.sqlite3")):
                print(f"Forçando reprocessamento: Removendo checkpoint antigo em {checkpoint_dir}")
                import shutil
                shutil.rmtree(checkpoint_dir) # Remove diretório antigo
                os.makedirs(checkpoint_dir, exist_ok=True) # Recria diretório vazio

            if os.path.exists(os.path.join(checkpoint_dir, "chroma.sqlite3")) and not self.force_reprocess:
                print(f"Carregando embeddings do schema do checkpoint: {checkpoint_dir}")
                return Chroma(
                    persist_directory=checkpoint_dir,
                    embedding_function=embedding_function
                )

            print(f"Gerando novos embeddings do schema em: {checkpoint_dir}")
            # A função extract_all_tables_info usará a format_table_info_for_embedding atualizada
            documents = self.schema_extractor.extract_all_tables_info()
            print(f"Total de documentos a serem incorporados: {len(documents)}")

            vectorstore = Chroma(
                persist_directory=checkpoint_dir,
                embedding_function=embedding_function
            )

            batch_size = 50
            # Filtra documentos vazios ou inválidos antes de adicionar
            valid_documents = [doc for doc in documents if doc and getattr(doc, 'page_content', None)]
            print(f"Documentos válidos para incorporação: {len(valid_documents)}")

            for i in range(0, len(valid_documents), batch_size):
                batch = valid_documents[i:i+batch_size]
                # Gera IDs únicos e robustos (evita problemas com caracteres especiais)
                ids = [f"doc_{hashlib.md5(str(i+j).encode()).hexdigest()}" for j in range(len(batch))]

                try:
                    vectorstore.add_documents(documents=batch, ids=ids)
                    print(f"Processado lote {i//batch_size + 1}/{(len(valid_documents)-1)//batch_size + 1} ({i+len(batch)}/{len(valid_documents)} documentos)")
                except Exception as add_doc_err:
                    print(f"ERRO ao adicionar lote de documentos ({i}-{i+len(batch)}): {add_doc_err}")
                    # Opcional: Tentar adicionar um por um para isolar o erro
                    # for k, doc_item in enumerate(batch):
                    #     try:
                    #         vectorstore.add_documents(documents=[doc_item], ids=[ids[k]])
                    #     except Exception as single_add_err:
                    #          print(f"--> Falha ao adicionar documento individual {ids[k]}: {single_add_err}")
                    #          print(f"    Conteúdo: {getattr(doc_item, 'page_content', 'N/A')[:200]}...") # Log do conteúdo problemático

            print("Embeddings do schema concluídos e salvos.")
            return vectorstore
        except Exception as e:
            print(f"Erro CRÍTICO ao criar/carregar embeddings do schema: {str(e)}")
            import traceback
            traceback.print_exc()
            # Fallback mais robusto
            fallback_dir = os.path.join(self.data_dir, 'checkpoints', 'fallback')
            os.makedirs(fallback_dir, exist_ok=True)
            print(f"Usando diretório de fallback: {fallback_dir}")
            try:
                # Tenta criar/carregar no fallback
                return Chroma(persist_directory=fallback_dir, embedding_function=self.embeddings)
            except Exception as fallback_e:
                print(f"ERRO CRÍTICO: Falha ao inicializar Chroma mesmo no diretório de fallback: {fallback_e}")
                raise fallback_e # Re-levanta a exceção se nem o fallback funcionar

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
    
    def find_relevant_tables(self, query: str, top_k: int = 7, structure_search_k: int = 20) -> List[str]:
        """
        Encontra tabelas relevantes para a consulta com busca aprimorada por estrutura.
        Args:
            query: A pergunta do usuário.
            top_k: O número máximo final de tabelas a retornar.
            structure_search_k: O número de documentos de estrutura a buscar inicialmente (ligeiramente aumentado).
        """
        tables = []
        # --- String de busca modificada ---
        search_query = f"Schema details relevant for the user query: {query}"
        logging.info(f"Buscando {structure_search_k} estruturas de tabela com query: '{search_query}'")
        # ---------------------------------
        try:
            # --- Busca Principal: Focada em Estrutura de Tabela ---
            docs_structure = self.vector_store.similarity_search(
                search_query, # Usa a query modificada
                k=structure_search_k,
                filter={"type": "table_structure"}
            )

            # Extrai nomes de tabelas únicos dos resultados da busca por estrutura
            for doc in docs_structure:
                table_name = doc.metadata.get("table_name")
                if table_name and table_name not in tables:
                    tables.append(table_name)
                    # Log de debug removido
                    # logging.debug(f"Tabela relevante encontrada (estrutura): {table_name} (Score: {doc.metadata.get('_score', 'N/A')})")
            logging.info(f"Encontradas {len(tables)} tabelas únicas na busca inicial por estrutura.")

            # --- Fallback (Opcional): Busca genérica se poucas tabelas encontradas ---
            if len(tables) < top_k // 2: # Se encontrou menos que metade do desejado
                logging.warning(f"Poucas tabelas encontradas ({len(tables)}). Realizando busca genérica adicional (k={top_k}).")
                # Usa a query original para o fallback genérico
                docs_generic = self.vector_store.similarity_search(query, k=top_k)
                for doc in docs_generic:
                    table_name = doc.metadata.get("table_name")
                    if table_name and table_name not in tables:
                        tables.append(table_name)
                        # Log de debug removido
                        # logging.debug(f"Tabela relevante encontrada (fallback genérico): {table_name} (Score: {doc.metadata.get('_score', 'N/A')})")
                logging.info(f"Total de tabelas após busca genérica: {len(tables)}.")

        except Exception as e:
            logging.error(f"ERRO durante a busca por tabelas relevantes: {e}", exc_info=True)
            # Em caso de erro, retorna uma lista vazia ou talvez as tabelas encontradas até agora
            # return tables[:top_k]

        # Limita ao número máximo final de tabelas
        final_tables = tables[:top_k]
        logging.info(f"Tabelas relevantes finais selecionadas: {', '.join(final_tables)}")
        return final_tables

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

    def _extract_tables_from_sql(self, sql: str) -> List[str]:
        """Extrai nomes de tabelas de uma string SQL usando regex (forma aprimorada)."""
        # Regex aprimorado:
        # - Procura por palavras após FROM ou JOIN
        # - Permite nomes qualificados (schema.table) ou entre aspas
        # - Tenta evitar capturar aliases comuns ou palavras-chave logo após o nome
        # - Ignora subconsultas entre parênteses logo após FROM/JOIN
        pattern = r'(?:FROM|JOIN)\s+((?:[\w"]+\.)?[\w"]+)(?:\s+(?:AS\s)?[\w"]+)?(?:\s+ON|\s+WHERE|\s+GROUP|\s+ORDER|\s+LIMIT|\s*$|\))'

        # Encontra todas as correspondências potenciais
        potential_tables = re.findall(pattern, sql, re.IGNORECASE | re.MULTILINE)

        # Filtra e limpa os nomes
        cleaned_tables = []
        # Lista de palavras-chave/funções comuns a ignorar (pode ser expandida)
        sql_keywords_to_ignore = {'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WHERE', 'GROUP', 'ORDER', 'BY', 'LIMIT', 'OFFSET', 'ON', 'AS', 'WITH', 'VALUES', 'SET', 'HAVING', 'UNION', 'ALL', 'DISTINCT', 'FROM', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER', 'FULL', 'CROSS', 'CURRENT_DATE', 'CURRENT_TIMESTAMP', 'NOW'}

        for table in potential_tables:
            # Remove aspas e espaços extras
            clean_name = table.strip('"').strip()
            # Verifica se não é uma palavra-chave SQL óbvia
            if clean_name.upper() not in sql_keywords_to_ignore and '.' not in clean_name: # Simplificação: Ignora nomes com ponto por enquanto (como aliases de coluna)
                cleaned_tables.append(clean_name)

        # Remove duplicatas mantendo a ordem
        unique_tables = list(dict.fromkeys(cleaned_tables))
        # Log de debug removido
        # logging.debug(f"Tabelas extraídas e filtradas do SQL do exemplo: {unique_tables}")
        return unique_tables

    def enhance_query_with_table_info(self, query: str, tables_to_include: List[str], selected_example_data: Optional[Dict] = None) -> str:
        """
        Enriquece a consulta com informações sobre tabelas especificadas, relações,
        e opcionalmente um exemplo few-shot.
        """
        print(f"DEBUG: Construindo prompt para tabelas: {tables_to_include}")
        # Coleta informações sobre estrutura das tabelas especificadas
        table_structures = []
        if not tables_to_include:
            print("AVISO: Nenhuma tabela especificada para incluir no prompt.")
        else:
            for table in tables_to_include:
                try:
                    # Usa o schema_extractor para obter info da tabela específica
                    table_info = self.schema_extractor.extract_table_structure_document(table)
                    table_structures.append(table_info.page_content)
                except Exception as e:
                    print(f"ERRO ao extrair estrutura da tabela '{table}' para o prompt: {e}")


        # Obtém relacionamentos APENAS entre as tabelas especificadas
        relationships = self.get_table_relationships(tables_to_include)
        relationship_info = []
        for table, related_tables in relationships.items():
            if related_tables:
                # Filtra para mostrar apenas relações dentro do conjunto fornecido
                valid_related = [rt for rt in related_tables if rt in tables_to_include]
                if valid_related:
                    rel_str = f"Tabela {table} relaciona-se com: {', '.join(valid_related)}"
                    relationship_info.append(rel_str)


        # Coleta amostras de dados APENAS para tabelas especificadas e importantes
        sample_data_info = []
        important_tables_in_set = [t for t in tables_to_include if self.schema_extractor._is_important_table(t)][:3]
        for table in important_tables_in_set:
            try:
                data_doc = self.schema_extractor.extract_table_data_document(table)
                if data_doc:
                    sample_data_info.append(data_doc.page_content)
            except Exception as e:
                print(f"ERRO ao extrair dados da tabela '{table}' para o prompt: {e}")


        # Contexto Odoo (sem alterações)
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

            Estrutura de produtos no Odoo:
            - product_template: contém informações básicas do produto como nome (name), descrição, etc.
            - product_product: representa variantes específicas de produtos e se relaciona com product_template através do campo product_tmpl_id
            - Para consultar o nome de produtos em sale_order_line, deve-se fazer JOIN entre sale_order_line, product_product e product_template

            Estrutura de vendas no Odoo:
            - sale_order: cabeçalho da venda com data (date_order), cliente (partner_id), etc.
            - sale_order_line: itens da venda com produto (product_id -> product_product), quantidade (product_uom_qty), valor unitário (price_unit), subtotal (price_subtotal)

            Estrutura de estoque no Odoo:
            - stock_warehouse: define os armazéns na empresa
            - stock_location: define localizações hierárquicas dentro dos armazéns (tipo=internal) e também localizações virtuais (clientes, fornecedores)
            - stock_move: registra movimentações de produtos entre localizações, com campos como product_id, product_uom_qty, location_id (origem), location_dest_id (destino)
            - stock_quant: registra a quantidade atual de cada produto em cada localização
            - stock_picking: representa transferências/remessas com um ou mais produtos
            - stock_picking_type: define tipos de operações de estoque (recebimento, entrega, transferência interna)
            - stock_inventory: usado para inventários físicos e ajustes de estoque

            Estrutura de compras no Odoo:
            - purchase_order: cabeçalho do pedido de compra com data (date_order), fornecedor (partner_id), moeda (currency_id), etc.
            - purchase_order_line: itens da compra com produto (product_id), quantidade (product_qty), preço unitário (price_unit), subtotal (price_subtotal)
            - purchase_requisition: requisição de compra (licitação) que pode gerar vários pedidos de compra
            - res_partner: para fornecedores, o campo supplier_rank indica a relevância como fornecedor
            - product_supplierinfo: contém informações específicas de fornecedores para produtos (preços, códigos de referência, tempo de entrega)
            - purchase_report: visão analítica das compras para relatórios

            Fluxo de aprovação de compras:
            - purchase_order.state: estados do pedido ('draft', 'sent', 'to approve', 'purchase', 'done', 'cancel')
            - purchase_order.approval_required: indica se o pedido precisa ser aprovado
            - purchase_order.user_id: usuário responsável pela compra
            - purchase_order.notes: notas internas da compra
            - purchase_order.date_planned: data planejada para recebimento

            Acordos com fornecedores:
            - purchase_order.origin: referência à origem do pedido (pode ser outro documento)
            - purchase_order.date_approve: data de aprovação da compra
            - purchase_order.fiscal_position_id: posição fiscal aplicada à compra
            - purchase_order.payment_term_id: condições de pagamento acordadas

            Estrutura de faturas no Odoo:
            - account_move: representa documentos contábeis incluindo faturas (invoice), pagamentos, lançamentos, etc.
            - O campo 'move_type' indica o tipo: 'out_invoice' (fatura de cliente), 'in_invoice' (fatura de fornecedor), 'out_refund' (devolução de cliente), 'in_refund' (devolução a fornecedor)
            - O campo 'state' indica o status: 'draft', 'posted', 'cancel', etc.
            - account_move_line: linhas dos lançamentos contábeis/faturas com produto, conta contábil, valores, etc.
            - account_payment: registra pagamentos de clientes e a fornecedores
            - account_journal: define os diários contábeis (vendas, compras, banco, caixa)

            Rastreabilidade de produtos:
            - stock_production_lot: define lotes/números de série para produtos rastreáveis
            - stock_move_line: detalha as movimentações com informações de lote/série através do campo lot_id

            Relacionamentos importantes entre módulos:
            - sale_order -> stock_picking: vendas geram remessas/entregas
            - stock_picking -> account_move: entregas podem gerar faturas
            - purchase_order -> stock_picking: compras geram recebimentos
            - purchase_order -> account_move: compras geram faturas de fornecedor
            - stock_picking -> stock_move: cada transferência tem uma ou mais movimentações de produtos
            - account_move -> purchase_order: faturas podem ser vinculadas a pedidos de compra via invoice_origin
            - purchase_order_line -> account_move_line: linhas de compra são vinculadas às linhas da fatura

            Campos de estado comuns:
            - sale_order.state: 'draft', 'sent', 'sale', 'done', 'cancel'
            - purchase_order.state: 'draft', 'sent', 'to approve', 'purchase', 'done', 'cancel'
            - stock_picking.state: 'draft', 'waiting', 'confirmed', 'assigned', 'done', 'cancel'
            - account_move.state: 'draft', 'posted', 'cancel'
            """

        # --- Formatar Exemplo Selecionado (se houver) ---
        selected_examples_str = ""
        if selected_example_data:
            try:
                question_text = selected_example_data.get('question', 'Pergunta Exemplo Ausente')
                sql_query = selected_example_data.get('query', 'SQL Exemplo Ausente')
                selected_examples_str = "Aqui está um exemplo altamente relevante de pergunta e SQL correto:\n\n"
                selected_examples_str += f"Pergunta Exemplo: {question_text}\nSQL Correto:\n```sql\n{sql_query}\n```\n\n"
                selected_examples_str += "---\n\n" # Separador
            except Exception as e:
                print(f"AVISO: Erro ao formatar exemplo: {e}")
        # --------------------------------------------

        # --- Cria prompt enriquecido ---
        # Instrução explícita para priorizar o exemplo, se fornecido
        example_priority_instruction = ""
        if selected_examples_str:
            example_priority_instruction = ("Use o exemplo SQL acima como guia principal, adaptando-o "
                                            "para a Pergunta do Usuário atual, especialmente quanto a detalhes "
                                            "como intervalos de tempo e valores. Use as informações de tabela abaixo "
                                            "para entender as colunas e relações das tabelas mencionadas no exemplo.\n\n")


        enhanced_query = f"""
{selected_examples_str}
{example_priority_instruction}
Contexto sobre o banco de dados (se aplicável):
{odoo_context}

Informações sobre as tabelas necessárias para esta consulta:

Estruturas das Tabelas:
{chr(10).join(table_structures) if table_structures else "Nenhuma informação de estrutura de tabela disponível."}

Relacionamentos entre tabelas relevantes:
{chr(10).join(relationship_info) if relationship_info else "Não foram identificados relacionamentos diretos entre estas tabelas."}

Amostras de dados de tabelas importantes (se disponíveis):
{chr(10).join(sample_data_info) if sample_data_info else "Nenhuma amostra de dados disponível."}

---
Pergunta do Usuário: {query}
---

Com base em TODAS as informações acima (especialmente o exemplo, se fornecido, e o contexto Odoo, schema, relações, amostras), gere a consulta SQL que melhor responde à Pergunta do Usuário.
Lembre-se de ajustar o SQL do exemplo para corresponder exatamente aos detalhes (datas, números) da pergunta do usuário.
Use joins quando apropriado e prefira JOINs explícitos (ex: INNER JOIN).
Inclua aliases de tabela para melhorar a legibilidade.
A consulta SQL final deve ser colocada dentro de um bloco ```sql ... ```.
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


    # --- MÉTODO query MODIFICADO ---
    def query(self, user_question: str) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Processa pergunta do usuário, usando exemplos para guiar a seleção de tabelas
        e o prompt, e fallback para busca no schema se nenhum exemplo for encontrado.
        """
        # Verifica cache (sem alterações)
        cached_data = self.get_query_from_cache(user_question)
        if cached_data:
            print("Usando resultado em cache")
            return cached_data['result'], cached_data['query']

        self.query_callback_handler.reset()

        tables_for_prompt = []
        selected_example = None
        example_sql = None

        # 1. Tenta selecionar exemplo relevante
        if self.example_selector:
            try:
                # Seleciona apenas o exemplo MAIS relevante (k=1 implicitamente aqui)
                # Se o seletor foi configurado com k>1, pegamos o primeiro/melhor
                selected_examples_metadata = self.example_selector.select_examples({"query": user_question})
                if selected_examples_metadata:
                    selected_example = selected_examples_metadata[0] # Pega o primeiro (mais similar)
                    print(f"--- Exemplo Selecionado para guiar tabelas ---")
                    print(f"  Q: {selected_example.get('question')}")
                    print(f"---------------------------------------------")
                    example_sql = selected_example.get('query')
            except Exception as e:
                print(f"AVISO: Erro ao tentar selecionar exemplo: {e}")

        # 2. Se um exemplo foi encontrado, extrai tabelas dele
        if example_sql:
            tables_for_prompt = self._extract_tables_from_sql(example_sql)
            if not tables_for_prompt:
                print("AVISO: Não foi possível extrair tabelas do SQL do exemplo selecionado. Tentando busca no schema.")
                example_sql = None # Reseta para forçar fallback
                selected_example = None

        # 3. Fallback: Se nenhum exemplo ou tabelas do exemplo encontrados, busca no schema
        if not tables_for_prompt:
            print("INFO: Nenhum exemplo relevante encontrado ou tabelas não extraídas. Usando busca vetorial no schema.")
            # Usa a função find_relevant_tables como antes
            tables_for_prompt = self.find_relevant_tables(user_question)
            print(f"Tabelas relevantes (fallback da busca no schema): {', '.join(tables_for_prompt)}")

        # 4. Constrói o prompt enriquecido com as tabelas determinadas e o exemplo (se houver)
        enhanced_question = self.enhance_query_with_table_info(
            user_question,
            tables_for_prompt,
            selected_example # Passa o dicionário do exemplo selecionado
        )

        # 5. Executa o agente (sem alterações)
        start_time = time.time()
        result = self.agent.invoke(
            {"input": enhanced_question},
            config={"callbacks": [self.query_callback_handler]}
        )
        exec_time = time.time() - start_time
        print(f"Tempo total de processamento: {exec_time:.2f}s")

        captured_query = self.query_callback_handler.get_captured_query()
        query_exec_time = self.query_callback_handler.get_execution_time()
        if query_exec_time:
            print(f"Tempo de execução da query: {query_exec_time:.2f}s")

        self.add_query_to_cache(user_question, result, captured_query)
        return result, captured_query            