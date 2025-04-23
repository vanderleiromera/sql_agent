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
        user_question = st.text_area(
            "O que você gostaria de saber?",
            height=100,
            placeholder="Ex: Quais são os 5 clientes que mais alugaram filmes?"
        )
        
        examples = [
            "Quais são os 5 filmes mais alugados?",
            "Qual a receita total por categoria de filme?",
            "Quais clientes estão com pagamentos em atraso?"
        ]
        
        st.write("Ou tente um exemplo:")
        for ex in examples:
            if st.button(ex):
                user_question = ex
                st.session_state.question = ex
        
        if st.button("Executar consulta", type="primary"):
            if not user_question:
                st.error("Por favor, digite uma pergunta!")
                return
            
            with st.spinner("Processando sua pergunta..."):
                # Encontrar tabelas relevantes
                relevant_tables = sql_agent.find_relevant_tables(user_question)
                st.session_state.tables = relevant_tables
                
                # Executar consulta
                result = sql_agent.query(user_question)
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