# Teste do QuerySQLCheckerTool
from modules.sql_agent import QuerySQLCheckerTool
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase

# Configuração básica para teste
engine = create_engine("sqlite:///:memory:")
db = SQLDatabase(engine=engine)
llm = ChatOpenAI(temperature=0)

# Cria o validador
checker = QuerySQLCheckerTool(llm=llm, db=db)

# Testa com uma consulta válida
valid_query = "SELECT * FROM users WHERE age > 18"
is_valid, result = checker.validate_query(valid_query)
print(f"Consulta válida: {is_valid}")
print(f"Resultado: {result}")

# Testa com uma consulta inválida (comando não permitido)
invalid_query = "DELETE FROM users WHERE id = 1"
is_valid, result = checker.validate_query(invalid_query)
print(f"Consulta inválida: {is_valid}")
print(f"Resultado: {result}")
