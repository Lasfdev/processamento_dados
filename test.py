import mlflow
import pandas as pd
import numpy as np

run_id = "ad90830b88834150be31325c529f3e41"
model_uri = f"runs:/{run_id}/modelo_pipeline"

# carrega o pipelie salvo
modelo_carregado = mlflow.pyfunc.load_model(model_uri)

# novos dados brutos 
novos_dados = pd.DataFrame({
    'idade': [28.0, np.nan, 52.0, 41.0, 19.0],
    'genero': ['Feminino', 'Masculino', np.nan, 'Feminino', 'Masculino'],
    'total_gasto': [150.50, 410.25, 95.00, 280.75, 512.30],
    'plano': ['Basico', 'Premium', 'Basico', 'Intermediario', 'Premium']
})


# o pipeline vai imputar, escalar, fazer o one-hot encoding e prever sozinho
previsoes = modelo_carregado.predict(novos_dados)
print("Previsões:", previsoes)