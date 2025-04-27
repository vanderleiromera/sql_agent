# Arquivo: modules/prompts.py
import os
import chromadb # Importar a biblioteca principal do Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# Prefixo de segurança para o Agente SQL
SAFETY_PREFIX = """IMPORTANTE: Você está interagindo com um banco de dados SQL. Sua única função é gerar consultas SQL do tipo SELECT para responder às perguntas dos usuários.
NUNCA gere comandos SQL que modifiquem dados ou o schema, como INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, etc.
Sempre gere apenas comandos SELECT.
Se a pergunta do usuário implicar uma modificação ou comando não permitido, informe que você só pode executar consultas SELECT para buscar informações."""

# --- Contexto Estático Específico do Odoo ---
# Adicione aqui dicas gerais e importantes sobre o seu schema Odoo. Mantenha conciso.
ODODO_STATIC_HINTS = """
Dicas importantes sobre Odoo:
- A tabela 'res_partner' contém clientes, fornecedores e contatos. Use 'is_company = TRUE' para filtrar apenas empresas. 'parent_id' indica a empresa mãe de um contato.
- Pedidos de venda estão em 'sale_order', faturas em 'account_move'. O status ('state') é crucial (ex: 'sale' para confirmado, 'draft' para rascunho).
- Produtos estão em 'product_template' (modelo geral) e 'product_product' (variante específica).
- `product_product` se relaciona com `product_template` através do campo `product_tmpl_id`.
- O estoque geralmente envolve 'stock_quant', que se relaciona com 'product_product'.
- Nomes de tabelas e colunas geralmente usam underscores (_) e são em minúsculas.
- Use IDs numéricos inteiros para relacionamentos (ex: partner_id, product_id, user_id).
"""
# --- Fim do Contexto Estático ---

# --- Constantes ---
# Use uma variável de ambiente ou um caminho fixo DENTRO do contêiner
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "/app/chroma_db")
CHROMA_COLLECTION_NAME = "odoo_sql_examples"

# Cria o diretório de persistência se não existir (importante para a primeira execução)
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# --- Configuração do Prompt Dinâmico Few-Shot ---

# 1. Exemplos Iniciais (Odoo)
examples = [
    # ... (cole a lista completa de exemplos aqui, como definida anteriormente) ...
    {
        "input": "Liste os nomes e emails de todos os parceiros (clientes/fornecedores).",
        "query": "SELECT name, email FROM res_partner;",
    },
    {
        "input": "Quais parceiros são empresas?",
        "query": "SELECT name FROM res_partner WHERE is_company = TRUE;",
    },
    {
        "input": "Encontre o parceiro com o email 'exemplo@mail.com'.",
        "query": "SELECT * FROM res_partner WHERE email = 'exemplo@mail.com';",
    },
    {
        "input": "Liste todos os produtos e seus preços de venda.",
        "query": "SELECT name, list_price FROM product_template;",
    },
    {
        "input": "Quantos produtos temos em estoque?",
        "query": """
            SELECT pt.name, SUM(sq.quantity) AS total_quantity
            FROM stock_quant sq
            JOIN product_product pp ON sq.product_id = pp.id
            JOIN product_template pt ON pp.product_tmpl_id = pt.id
            WHERE sq.location_id IN (SELECT id FROM stock_location WHERE usage = 'internal')
            GROUP BY pt.name;
            """,
    },
    {
        "input": "Quais são as categorias de produtos?",
        "query": "SELECT name FROM product_category;",
    },
     {
        "input": "Liste todos os pedidos de venda confirmados.",
        "query": "SELECT name, partner_id, date_order FROM sale_order WHERE state = 'sale';",
    },
     {
        "input": "Quem são os 5 principais clientes por valor total de pedidos de venda?",
        "query": """
            SELECT rp.name, SUM(so.amount_total) as total_spent
            FROM sale_order so
            JOIN res_partner rp ON so.partner_id = rp.id
            WHERE so.state = 'sale'
            GROUP BY rp.name
            ORDER BY total_spent DESC
            LIMIT 5;
            """,
    },
]

# --- Inicialização do ChromaDB com Persistência ---
vectorstore = None
embeddings = None
try:
    embeddings = OpenAIEmbeddings()

    # Inicializa o cliente Chroma apontando para o diretório persistente
    # DeprecationWarning: Deprecated class Chroma. Use from chromadb.Client object (https://docs.trychroma.com/getting-started)
    # Usando a abordagem recomendada com o cliente ChromaDB:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Tenta obter a coleção. Se não existir, será criada depois.
    try:
        collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)
        print(f"INFO: Coleção Chroma '{CHROMA_COLLECTION_NAME}' carregada do diretório '{CHROMA_PERSIST_DIR}'. Count: {collection.count()}")
        # Se a coleção existe mas está vazia (improvável, mas possível), recriamos
        if collection.count() == 0:
             print(f"AVISO: Coleção Chroma '{CHROMA_COLLECTION_NAME}' encontrada mas vazia. Recriando...")
             chroma_client.delete_collection(name=CHROMA_COLLECTION_NAME) # Remove a vazia
             raise chromadb.exceptions.CollectionNotFoundError # Força a criação abaixo

        # Cria a interface LangChain VectorStore usando a coleção existente
        vectorstore = Chroma(
            client=chroma_client,
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR # Ainda útil para LangChain saber onde está
        )

    except chromadb.exceptions.CollectionNotFoundError:
        print(f"INFO: Coleção Chroma '{CHROMA_COLLECTION_NAME}' não encontrada. Criando e populando...")
        # Coleção não existe, cria usando from_examples (que usa o cliente implicitamente se disponível)
        # ou diretamente com o cliente. LangChain recomenda passar o cliente.
        vectorstore = Chroma.from_examples(
            client=chroma_client, # Garante que use o cliente persistente
            examples=examples,
            embedding=embeddings,
            ids=[f"example_{i}" for i, _ in enumerate(examples)],
            collection_name=CHROMA_COLLECTION_NAME,
            persist_directory=CHROMA_PERSIST_DIR # Informa ao LangChain onde persistir
        )
        # Não é mais necessário chamar vectorstore.persist() explicitamente com PersistentClient
        print(f"INFO: Coleção Chroma '{CHROMA_COLLECTION_NAME}' criada e persistida em '{CHROMA_PERSIST_DIR}'.")

except Exception as e:
    print(f"ERRO CRÍTICO ao inicializar embeddings ou ChromaDB em prompts.py: {e}")
    # Tratar o erro como apropriado. Sem vectorstore, o fallback será usado.
    embeddings = None
    vectorstore = None

# --- Fim da Inicialização do ChromaDB ---

# 3. Seletor de Similaridade
if vectorstore:
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=5,
        input_keys=["input"],
    )
else:
    example_selector = None # Falha na inicialização

# 4. Prompt para Exemplos Individuais
example_prompt = PromptTemplate.from_template(
    "Pergunta do Usuário: {input}\nConsulta SQL: {query}"
)

# 5. Prompt Few-Shot Dinâmico Final
#    Nota: O prefixo combina o SAFETY_PREFIX com as instruções SQL.
if example_selector:
    ODODO_FEW_SHOT_PROMPT = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=f"""{SAFETY_PREFIX}

{ODODO_STATIC_HINTS}

Você é um especialista em PostgreSQL e Odoo. Dada uma pergunta de entrada, crie uma consulta PostgreSQL sintaticamente correta para executar contra um banco de dados Odoo.
A menos que especificado de outra forma, não retorne mais do que {{top_k}} linhas.

Você só pode usar as tabelas listadas abaixo. Não use tabelas que não estejam na lista. Tenha cuidado para não consultar colunas que não existem nas tabelas. Preste atenção à capitalização das tabelas e colunas.

Aqui estão as informações das tabelas relevantes:
{{table_info}}

Abaixo estão alguns exemplos de perguntas e suas consultas SQL correspondentes:""",
        suffix="Pergunta do Usuário: {input}\nConsulta SQL:",
        input_variables=["input", "top_k", "table_info"],
    )
else:
    # Fallback: Crie um prompt simples sem few-shot se o seletor falhou
    print("AVISO: Falha ao criar o seletor de exemplos. Usando prompt padrão sem few-shot.")
    # Você pode definir um PromptTemplate básico aqui como alternativa
    # Exemplo:
    # ODODO_FEW_SHOT_PROMPT = PromptTemplate.from_template(
    #     f"{SAFETY_PREFIX}\n\nInstruções SQL básicas... {{table_info}}\n\nPergunta: {{input}}\nSQL:"
    # )
    # Ou simplesmente definir como None e tratar isso no sql_agent.py
    ODODO_FEW_SHOT_PROMPT = None


# --- Fim da Configuração do Prompt Dinâmico ---

# Você pode adicionar outros prompts aqui no futuro, por exemplo:
# ANOTHER_PROMPT = """..."""
