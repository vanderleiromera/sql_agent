"""
Módulo para otimização de consultas SQL em bancos de dados Odoo.

Este módulo fornece funcionalidades especializadas para otimizar consultas SQL
em bancos de dados Odoo, especialmente para bancos grandes com muitas tabelas e relações.
Implementa estratégias como:
- Identificação de tabelas relevantes
- Enriquecimento de contexto com relações importantes
- Execução com timeout para garantir performance
- Cache de resultados para consultas frequentes
"""
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
import hashlib
import time
import datetime
import threading
import re
from .context import ODOO_CONTEXT

class OdooQueryOptimizer:
    """
    Otimizador de consultas SQL para bancos de dados Odoo.
    
    Esta classe implementa funcionalidades avançadas para otimizar consultas SQL
    em bancos de dados Odoo, reduzindo o contexto necessário e melhorando a performance.
    
    Attributes:
        agent: Agente LangChain para executar consultas
        schema_extractor: Extrator de esquema para obter metadados do banco
        query_cache: Cache para armazenar resultados de consultas
        config: Configurações do otimizador
    """
    
    def __init__(self, agent=None, schema_extractor=None, callback_handler=None, config=None):
        """
        Inicializa o otimizador de consultas.
        
        Args:
            agent: Agente LangChain para executar as consultas
            schema_extractor: Extrator de esquema do banco
            callback_handler: Handler para capturar consultas e métricas
            config: Configurações customizadas para o otimizador
        """
        self.agent = agent
        self.schema_extractor = schema_extractor
        self.callback_handler = callback_handler
        self.config = config
        self.query_cache = {}
        self._table_count = None
        self._odoo_module_count = None
        self.odoo_context = None
    
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
            # 1. Identificar tabelas relevantes (usando método externo)
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
    
    def find_relevant_tables(self, user_question: str) -> List[str]:
        """
        Encontra tabelas relevantes para a pergunta do usuário.
        
        Esta função deve ser implementada pelo cliente que usar esta classe.
        O método atual apenas retorna uma lista vazia e deve ser sobrescrito
        ou implementado pelo objeto que instancia esta classe.
        
        Args:
            user_question: Pergunta do usuário em linguagem natural
            
        Returns:
            Lista de nomes de tabelas relevantes
        """
        # Implementação dummy a ser sobrescrita pelo usuário desta classe
        return []
    
    def enhance_query_with_table_info(self, user_question: str, tables: List[str]) -> str:
        """
        Enriquece a consulta com informações das tabelas.
        
        Esta função deve ser implementada pelo cliente que usar esta classe.
        O método atual apenas retorna a própria pergunta e deve ser sobrescrito
        ou implementado pelo objeto que instancia esta classe.
        
        Args:
            user_question: Pergunta do usuário em linguagem natural
            tables: Lista de tabelas relevantes
            
        Returns:
            Consulta enriquecida com informações das tabelas
        """
        # Implementação dummy a ser sobrescrita pelo usuário desta classe
        return user_question
    
    def get_table_relationships(self, table_names: List[str]) -> Dict[str, List[str]]:
        """
        Obtém relacionamentos entre tabelas.
        
        Esta função deve ser implementada pelo cliente que usar esta classe.
        O método atual apenas retorna um dicionário vazio e deve ser sobrescrito
        ou implementado pelo objeto que instancia esta classe.
        
        Args:
            table_names: Lista de nomes de tabelas
            
        Returns:
            Dicionário com tabelas e suas relações
        """
        # Implementação dummy a ser sobrescrita pelo usuário desta classe
        return {}
        
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
                        if not rel_table.startswith("ir_") and hasattr(self.schema_extractor, '_is_important_table') and self.schema_extractor._is_important_table(rel_table):
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
        # Resetar callback handler
        if self.callback_handler:
            self.callback_handler.reset()
        
        try:
            # Variável para armazenar o resultado
            result_container = {"result": None, "done": False}
            
            # Função que executa a consulta
            def execute_query():
                try:
                    result = self.agent.run(enhanced_query, callbacks=[self.callback_handler] if self.callback_handler else None)
                    
                    # Capturar informações do callback se disponível
                    sql_query = None
                    execution_time = None
                    if self.callback_handler:
                        sql_query = self.callback_handler.get_captured_query()
                        execution_time = self.callback_handler.exec_time
                    
                    result_container["result"] = {
                        "success": True,
                        "result": result,
                        "sql_query": sql_query,
                        "execution_time": execution_time
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
        if hasattr(self, '_table_count') and self._table_count is not None:
            return self._table_count > 200
        
        # Caso contrário, contamos as tabelas
        try:
            if self.schema_extractor and hasattr(self.schema_extractor, 'get_all_tables'):
                tables = self.schema_extractor.get_all_tables()
                self._table_count = len(tables)
                
                # Verificar contagem de módulos Odoo também
                odoo_modules = set()
                for table in tables:
                    if hasattr(self.schema_extractor, '_infer_odoo_module'):
                        module = self.schema_extractor._infer_odoo_module(table)
                        if module:
                            odoo_modules.add(module)
                
                self._odoo_module_count = len(odoo_modules)
                
                # Consideramos grande se tiver muitas tabelas OU muitos módulos
                return self._table_count > 200 or self._odoo_module_count > 8
            return False
                
        except Exception as e:
            print(f"Erro ao determinar tamanho do banco: {str(e)}")
            # Em caso de erro, assumimos banco normal
            return False
