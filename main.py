# %%
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

# carregando o dataset
# %%
df = pd.read_csv("/home/lais/Projetos/portfolio/processamento_dados/data/data.csv",sep=",")
df

# %%
# Definindo o nome do experimento 
mlflow.set_experiment("previsao_churn")

# Ativando o log automático para scikit-learn
mlflow.sklearn.autolog()

# %%

