# Arquivo: modules/prompts.py
import os
import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.schema import Document

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
# Diretório base mapeado pelo Docker Volume
BASE_DATA_DIR = "/app/data"
# Subdiretório específico para os exemplos few-shot
CHROMA_FEW_SHOT_DIR = os.path.join(BASE_DATA_DIR, "few_shot_examples")
CHROMA_COLLECTION_NAME = "odoo_sql_examples"

# Cria o diretório de persistência específico se não existir
os.makedirs(CHROMA_FEW_SHOT_DIR, exist_ok=True)

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

# --- Inicialização do VectorStore e Seletor ---
vectorstore = None
example_selector = None
embeddings = None

try:
    embeddings = OpenAIEmbeddings()
    chroma_client = chromadb.PersistentClient(path=CHROMA_FEW_SHOT_DIR)
    collection = None

    # Tenta obter a coleção
    try:
        collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)
        print(f"INFO: Coleção Chroma '{CHROMA_COLLECTION_NAME}' encontrada em '{CHROMA_FEW_SHOT_DIR}'. Verificando contagem...")
    except Exception:
        print(f"AVISO: Coleção '{CHROMA_COLLECTION_NAME}' não encontrada. Será criada.")
        collection = None

    # Verifica se vazia ou não encontrada
    if collection and collection.count() == 0:
        print(f"AVISO: Coleção '{CHROMA_COLLECTION_NAME}' encontrada vazia. Recriando...")
        chroma_client.delete_collection(name=CHROMA_COLLECTION_NAME)
        collection = None

    # Cria/Recria se necessário, usando from_texts
    if collection is None:
        print(f"INFO: Criando/Recriando coleção '{CHROMA_COLLECTION_NAME}' com from_texts...")
        input_texts = [example["input"] for example in examples]
        metadatas = [{"query": example["query"]} for example in examples]
        ids = [f"example_{i}" for i, _ in enumerate(examples)]

        vectorstore = Chroma.from_texts(
            client=chroma_client,
            texts=input_texts,
            embedding=embeddings,
            metadatas=metadatas,
            ids=ids,
            collection_name=CHROMA_COLLECTION_NAME,
            persist_directory=CHROMA_FEW_SHOT_DIR
        )
        print(f"INFO: Coleção criada/recriada. Count: {vectorstore._collection.count()}")
    else:
        # Carrega a vectorstore existente
        print(f"INFO: Carregando coleção existente '{CHROMA_COLLECTION_NAME}'. Count: {collection.count()}")
        vectorstore = Chroma(
            client=chroma_client,
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_FEW_SHOT_DIR
        )

    # Criando um seletor personalizado que adapta os documentos para o formato esperado
    class CustomExampleSelector(SemanticSimilarityExampleSelector):
        def select_examples(self, input_variables):
            try:
                # First try getting examples through parent class
                docs = super().select_examples(input_variables)
                
                # Handle both Document objects and dicts
                converted_examples = []
                for doc in docs:
                    if hasattr(doc, 'page_content'):
                        # It's a Document object
                        example = {
                            "input": doc.page_content,
                            "query": doc.metadata.get("query", "")
                        }
                    elif isinstance(doc, dict):
                        # It's already a dict
                        example = {
                            "input": doc.get("input", ""),
                            "query": doc.get("query", "")
                        }
                    else:
                        continue  # Skip invalid formats
                        
                    converted_examples.append(example)
                    
                return converted_examples
                
            except Exception as e:
                print(f"Error in example selection: {e}")
                # Return empty list as fallback
                return []

    # Inicializa o seletor personalizado
    if vectorstore:
        example_selector = CustomExampleSelector(
            vectorstore=vectorstore,
            k=5,
            input_keys=["input"]
        )
        print("INFO: Seletor de exemplos dinâmicos personalizado inicializado com vectorstore existente.")

except Exception as e:
    print(f"ERRO CRÍTICO ao inicializar embeddings, ChromaDB ou Seletor em prompts.py: {e}")
    import traceback
    traceback.print_exc()
    embeddings = None
    vectorstore = None
    example_selector = None

# --- Fim da Inicialização ---

# 4. Prompt para exemplo individual (agora esperando as chaves corretas)
example_prompt = PromptTemplate.from_template(
    "Pergunta do Usuário: {input}\nConsulta SQL: {query}"
)

# 5. Prompt Few-Shot Dinâmico Final
ODODO_FEW_SHOT_PROMPT = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=f"""{SAFETY_PREFIX}

{ODODO_STATIC_HINTS}

Você é um especialista em PostgreSQL e Odoo. Dada uma pergunta de entrada, crie uma consulta PostgreSQL sintaticamente correta para executar contra um banco de dados Odoo.
A menos que especificado de outra forma, não retorne mais do que 5 linhas.

Você só pode usar as tabelas listadas abaixo. Não use tabelas que não estejam na lista. Tenha cuidado para não consultar colunas que não existem nas tabelas. Preste atenção à capitalização das tabelas e colunas.""",
    suffix="""
INFORMAÇÕES DAS TABELAS:
{table_info}

Pergunta: {input}
Consulta SQL:""",
    input_variables=["input", "table_info"],
    partial_variables={"top_k": 5}
)

if not ODODO_FEW_SHOT_PROMPT:
    print("AVISO: Falha ao criar o seletor de exemplos ou o prompt final. Usando prompt padrão sem few-shot.")
    # Corrigindo o fallback - removendo input_variables explicito
    DEFAULT_PROMPT_STR = f"{SAFETY_PREFIX}\n\nInstruções: Gere uma consulta SQL SELECT para responder à pergunta baseado nas tabelas abaixo.\n\nTabelas:\n{{table_info}}\n\nPergunta: {{input}}\nConsulta SQL:"
    ODODO_FEW_SHOT_PROMPT = PromptTemplate.from_template(DEFAULT_PROMPT_STR)  # Removido input_variables


# --- Fim da Configuração do Prompt Dinâmico ---

# Você pode adicionar outros prompts aqui no futuro, por exemplo:
# ANOTHER_PROMPT = """..."""