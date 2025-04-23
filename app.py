# arquivo: app.py
import streamlit as st
import os
from dotenv import load_dotenv
from modules.sql_agent import DVDRentalTextToSQL
from modules.config import Config

# Carrega variáveis de ambiente
load_dotenv()

# Configuração da página Streamlit
st.set_page_config(
    page_title="SQL Agent App",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def initialize_sql_agent(force_reprocess=False):
    """Inicializa o agente SQL (cache para não reinicializar a cada interação)"""
    config = Config()
    return DVDRentalTextToSQL(config.db_uri, use_checkpoint=True, force_reprocess=force_reprocess)

def main():
    st.title("🔍 Consultas em Linguagem Natural para SQL")
    st.subheader("Transforme perguntas em consultas SQL")
    
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
            placeholder="Ex: Quais são os 5 clientes que mais alugaram filmes?",
            key="user_question_area" # Adiciona uma chave para estabilidade
        )
        # Atualiza o session_state se o usuário digitar manualmente
        st.session_state.question = user_question_input

        examples = [
            "Quais são os 5 filmes mais alugados?",
            "Qual a receita total por categoria de filme?",
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
                result = sql_agent.query(current_question)
                st.session_state.result = result
    
    with col2:
        st.header("Resultados")
        
        # Mostrar tabelas relevantes
        if "tables" in st.session_state:
            st.subheader("Tabelas identificadas")
            st.write(", ".join(st.session_state.tables))
        
        # Mostrar resultados
        if "result" in st.session_state:
            result = st.session_state.result
            
            st.subheader("Consulta SQL")
            for action in result.get("intermediate_steps", []):
                if isinstance(action, tuple) and len(action) >= 2:
                    tool = action[0]
                    if tool.name == "query_sql_db":
                        st.code(tool.args.get("query"), language="sql")
            
            st.subheader("Resultados")
            if "output" in result:
                try:
                    # Tenta converter para dataframe se possível
                    import pandas as pd
                    if isinstance(result["output"], str) and result["output"].strip().startswith("|"):
                        # Resultado em formato markdown table
                        st.write(result["output"])
                    else:
                        st.write(result["output"])
                except:
                    st.write(result["output"])
            else:
                st.info("Nenhum resultado retornado")
            
            # Explicação adicional
            st.subheader("Explicação")
            explanation = result.get("output", "")
            if isinstance(explanation, str):
                st.write(explanation)

if __name__ == "__main__":
    main()