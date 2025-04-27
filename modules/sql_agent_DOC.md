# Módulo SQL Agent - Documentação Técnica

## Visão Geral
Módulo para conversão de linguagem natural em queries SQL, otimizado para bancos Odoo. Principais funcionalidades:

- Extração automática de esquema de banco de dados
- Geração de embeddings para busca semântica
- Processamento seguro de queries naturais
- Integração com modelos LLM da OpenAI
- Cache de queries e resultados

```mermaid
graph TD
    A[Consulta do Usuário] --> B{Encontra Tabelas Relevantes}
    B --> C[Enriquece Consulta com Metadados]
    C --> D[Gera SQL via LLM]
    D --> E[Executa no Banco]
    E --> F[Retorna Resultado]
    F --> G[Armazena em Cache]
```

## Classes Principais

### `SchemaExtractor`
**Responsabilidade:** Extração e processamento de metadados do banco

**Métodos Chave:
- `generate_rich_table_info()`: Estrutura metadados completos da tabela
- `extract_all_tables_info()`: Processa todas as tabelas do banco
- `_identify_table_category()`: Classifica tabelas por uso (negócio, técnica, etc)

**Exemplo de Uso:
```python
extractor = SchemaExtractor("postgresql://user:pass@host/db")
table_info = extractor.generate_rich_table_info("res_partner")
```

### `OdooTextToSQL`
**Responsabilidade:** Processamento central de queries naturais

**Componentes:
- Vector Store: ChromaDB para embeddings de esquema
- LLM: Modelo GPT-4 para geração de SQL
- Cache: Dupla camada (SQLAlchemy + memória)

**Fluxo Principal:
1. Recebe query natural
2. Busca tabelas relevantes via embeddings
3. Enriquece query com metadados
4. Gera SQL via LLM
5. Executa e retorna resultados

**Exemplo:
```python
agent = OdooTextToSQL("postgresql://user:pass@host/odoo")
result, query = agent.query("Liste os 10 clientes que mais compraram")
```

### `SQLQueryCaptureCallback`
**Monitoramento:** Captura queries SQL geradas e métricas

**Dados Coletados:
- Query SQL executada
- Tempo de execução
- Erros ocorridos

**Uso:
```python
callback = SQLQueryCaptureCallback()
agent.agent.callbacks = [callback]
```

## Configurações Importantes

### Tabelas Odoo
```python
ODOO_COMPLEX_TABLES = {
    'ir_filters': {'skip_sample_data': True},
    'ir_actions': {'skip_sample_data': True}
}

PROBLEMATIC_TABLES = {
    'ir_translation': {'skip': True},
    'stock_production_lot': {'max_samples': 5}
}
```

### Mapeamento de Módulos
```python
ODOO_MODULE_MAP = {
    "res": "base",
    "product": "product",
    "account": "accounting"
}
```

## Segurança
- Validação de queries SELECT antes da execução
- Sanitização de inputs via `SAFETY_PREFIX`
- Isolamento de dados sensíveis em cache

## Exemplo Completo
```python
from modules.sql_agent import OdooTextToSQL

# Configuração
db_uri = "postgresql://user:pass@prod-db/odoo"
query = "Clientes com mais vendas em 2024"

# Execução
agent = OdooTextToSQL(db_uri)
result, sql_query = agent.query(query)

print(f"SQL Gerado: {sql_query}")
print(f"Resultado: {result}")
```

## Monitoramento
Métricas coletadas via callback:
- Tempo de resposta total
- Tempo de execução do SQL
- Taxa de cache hit
- Erros por tipo de operação
