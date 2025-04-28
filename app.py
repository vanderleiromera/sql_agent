# arquivo: app.py
import streamlit as st
import os
from dotenv import load_dotenv
from modules.sql_agent import OdooTextToSQL
from modules.config import Config
import pandas as pd
import json
import sys

# Adiciona o diretório pai ao path para importar módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Carrega variáveis de ambiente
load_dotenv()

# Configuração da página Streamlit
st.set_page_config(
    page_title="Odoo ERP SQL Agent",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def initialize_sql_agent(force_reprocess=False):
    """Inicializa o agente SQL (cache para não reinicializar a cada interação)"""
    config = Config()
    return OdooTextToSQL(config.db_uri, use_checkpoint=True, force_reprocess=force_reprocess)

def main():
    st.title("🤖 Agente SQL para Consulta de Odoo ERP")
    st.write("Faça perguntas em linguagem natural sobre os dados de Odoo ERP.")
    
    # Sidebar com informações
    with st.sidebar:
        st.header("Sobre")
        st.write("""
        Este aplicativo usa LangChain e LLMs para converter 
        perguntas em linguagem natural para consultas SQL.
        """)
        
        st.header("Status")
        db_status = st.empty()

        st.header("Opções")
        force_reprocess = st.checkbox("Forçar reprocessamento do schema", value=False)
    
    # Inicializa o agente
    try:
        sql_agent = initialize_sql_agent(force_reprocess=force_reprocess)
        db_status.success("✅ Conectado ao banco de dados")
    except Exception as e:
        db_status.error(f"❌ Erro ao conectar: {str(e)}")
        st.error("Verifique suas configurações no arquivo .env")
        return
    
    # Interface principal
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Faça sua pergunta")

        # Inicializa a chave 'question' no session_state se não existir
        if 'question' not in st.session_state:
            st.session_state.question = ""

        # Usa st.session_state.question para controlar o valor do text_area
        # O valor retornado pelo text_area atualiza user_question_input
        user_question_input = st.text_area(
            "O que você gostaria de saber?",
            value=st.session_state.question,  # Vincula ao session_state
            height=100,
            placeholder="Ex: Qual total de vendas do ultimo 3 meses?",
            key="user_question_area" # Adiciona uma chave para estabilidade
        )
        # Atualiza o session_state se o usuário digitar manualmente
        st.session_state.question = user_question_input
        
        examples = [
            "Qual produto mais vendido em março, 2025? valor e nome",
            "Total de vendas em 2024, mês a mês?",
            "Liste os 10 produtos mais vendidos, em ordem de valor?"
        ]
        
        st.write("Ou tente um exemplo:")
        # Itera sobre os exemplos para criar os botões
        for i, ex in enumerate(examples):
            # Usa uma chave única para cada botão
            if st.button(ex, key=f"example_{i}"):
                # Atualiza o session_state com o exemplo clicado
                st.session_state.question = ex
                # Força um rerun para atualizar o text_area imediatamente
                st.rerun()
        
        if st.button("Executar consulta", type="primary"):
            if not st.session_state.question:
                st.error("Por favor, digite uma pergunta ou selecione um exemplo!")
                return

            with st.spinner("Processando sua pergunta..."):
                current_question = st.session_state.question
                # O método query agora lida internamente com tabelas e formatação
                # agent_result é o dicionário {'output': <query_sql>}
                # captured_sql_query é a mesma query (se capturada pelo callback)
                agent_result, captured_sql_query = sql_agent.query(current_question)

                # Armazenamos o resultado principal
                st.session_state.result = agent_result
                # Armazenamos a query capturada (pode ser útil para logs/depuração)
                # ou podemos usar diretamente agent_result['output']
                st.session_state.sql_query = agent_result.get("output") # Usar a query do resultado principal

                # Limpa 'tables' se existir, pois não é mais usado diretamente aqui
                if 'tables' in st.session_state:
                    del st.session_state['tables']

    with col2:
        st.header("Resposta do Agente")

        # Exibe a query SQL se ela existir no resultado
        if "result" in st.session_state and isinstance(st.session_state.result, dict) and st.session_state.result.get("output"):
            st.subheader("Consulta SQL Gerada")
            # Verifica se a saída é uma string (a query) antes de exibi-la com st.code
            sql_output = st.session_state.result["output"]
            if isinstance(sql_output, str):
                # Verifica se não é uma mensagem de erro
                if not sql_output.startswith("Erro ao gerar consulta SQL:"):
                    st.code(sql_output, language="sql")
                    st.divider()
                    st.subheader("Execução e Interpretação")
                    st.warning("A execução direta da consulta no banco de dados e a interpretação da resposta ainda precisam ser implementadas.")
                    # Aqui você adicionaria a lógica para realmente executar a query `sql_output`
                    # usando sql_agent.db.run(sql_output) ou similar e exibir os resultados.
                else:
                    # Exibe a mensagem de erro que veio do agente
                    st.error(sql_output)

            else:
                # Caso inesperado onde 'output' não é string
                st.error(f"Formato inesperado no resultado: {sql_output}")

        elif "result" in st.session_state: # Se houve execução mas sem 'output'
            st.info("Não foi possível gerar ou exibir a consulta SQL.")

        # Removido o bloco que exibia o resultado interpretado, pois agora o foco é gerar a query.
        # A execução e interpretação seriam o próximo passo.

        else:
            st.write("Aguardando a execução de uma consulta...")

if __name__ == "__main__":
    main()