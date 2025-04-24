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
            "Qual produto mais vendido em março, 2025? filte em sale_order e product_product, valor e nome",
            "Total de vendas produtos default code '1290' em valor no ano de 2024?",
            "Quais clientes estão com pagamentos em atraso?"
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
            # Verifica se há algo no session_state.question (seja de exemplo ou digitado)
            if not st.session_state.question:
                st.error("Por favor, digite uma pergunta ou selecione um exemplo!")
                return # Não use st.stop() aqui, apenas retorne
            
            with st.spinner("Processando sua pergunta..."):
                # Usa a pergunta do session_state
                current_question = st.session_state.question
                # Encontrar tabelas relevantes
                relevant_tables = sql_agent.find_relevant_tables(current_question)
                st.session_state.tables = relevant_tables
                
                # Executar consulta
                agent_result, captured_sql_query = sql_agent.query(current_question)
                st.session_state.result = agent_result
                st.session_state.sql_query = captured_sql_query
                
                # Limpa tabelas relevantes antigas se existirem
                if 'tables' in st.session_state:
                    del st.session_state['tables']
                
                # Atualiza tabelas relevantes (opcional, pode vir do agente se modificado)
                # relevant_tables = sql_agent.find_relevant_tables(current_question)
                # st.session_state.tables = relevant_tables
    
    with col2:
        st.header("Resposta do Agente")
        
        # Exibe a query SQL se foi capturada
        if "sql_query" in st.session_state and st.session_state.sql_query:
            st.subheader("Consulta SQL Gerada")
            st.code(st.session_state.sql_query, language="sql")
        elif "result" in st.session_state: # Exibe info apenas se houve uma execução
            st.subheader("Consulta SQL Gerada")
            st.info("Consulta SQL não foi capturada pelo callback nesta execução.")
        
        st.divider() # Separador

        # Exibe o resultado final do agente
        if "result" in st.session_state:
            st.subheader("Resultados")
            result_data = st.session_state.result
            final_answer = "Erro ao obter 'output'"

            if isinstance(result_data, dict):
                final_answer = result_data.get("output", "Chave 'output' não encontrada no resultado.")
            elif isinstance(result_data, str):
                final_answer = result_data
            else:
                final_answer = str(result_data)

            # Tenta exibir o resultado final (geralmente texto interpretado)
            st.write(final_answer)

            # Opcional: Exibir tabelas relevantes se armazenadas
            # if "tables" in st.session_state and st.session_state.tables:
            #    st.subheader("Tabelas Relevantes Identificadas")
            #    st.write(", ".join(st.session_state.tables))

        else:
            st.write("Aguardando a execução de uma consulta...")

if __name__ == "__main__":
    main()