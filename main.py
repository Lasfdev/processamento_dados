# %%
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


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
# Definindo as features e a variavel label 
features = ["idade","genero","total_gasto","plano"]
X = df[features]

# %%
y = df['churn']


# %%
# Diviir o dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# %%
# Definindo as etapas de pré-processamento (Pipeline)
colunas_numericas = ["idade","total_gasto"]
colunas_categoricas = ["genero", "plano"]

# %%
# Transformando os dados numéricos: preenche nulos com a média e padroniza
transformador_numerico = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])



# %%
# Transformando os dados categóricos: preenche nulos e aplica One-Hot Encoding
transformador_categorico = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# %%
# Unindo os transformadores com ColumnTransformer
preprocessador = ColumnTransformer(
    transformers = [
        ('num', transformador_numerico, colunas_numericas),
        ('cat', transformador_categorico, colunas_categoricas)
    ]
)

# %%
# Definindo os hiperparametros e criando Pipeline final
max_iter = 1000
random_state = 42
class_weight='balanced'
solver='liblinear'

pipeline_completo = Pipeline(steps=[
    ('preprocessamento', preprocessador),
    ('classificador', LogisticRegression(max_iter=max_iter, random_state=random_state, class_weight=class_weight,solver=solver))
])

# %%
with mlflow.start_run(run_name="LogistcRegression_com_preprocessamento"):
    
    # Treina o pipeline inteiro
    pipeline_completo.fit(X_train, y_train)

    # Faz previsões
    y_pred = pipeline_completo.predict(X_test)

    # Calcula métricas
    acuracia = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    # matrix = confusion_matrix(y_test,y_pred)
    # classification = classification_report(y_test,y_pred)

    # Log de Hiperparâmetros no MLflow
    mlflow.log_param("max_iter",max_iter)
    mlflow.log_param("random_state",random_state)
    mlflow.log_param("class_weight",class_weight)
    mlflow.log_param("solver",solver)
    mlflow.log_param("imputer_num_strategy", "mean")

    # Log de Métricas no Mlflow
    mlflow.log_metric("accuracy",acuracia)
    mlflow.log_metric("f1_score", f1)
    # mlflow.log_metric("classification_report",classification)
    # mlflow.log_metric("confusion_matrix",matrix)

    # Salva o PIPELINE INTEIRO como o modelo no Mlflow
    # O input_example ajuda o Mlflow a inferir o esquema (tipo de dados)
    mlflow.sklearn.log_model(
        sk_model = pipeline_completo,
        artifact_path="modelo_pipeline",
        input_example=X_train.iloc[:2],
        skops_trusted_types=["numpy.dtype"]
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
    mlflow.log_figure(fig, "matriz_confusao.png")
    plt.close(fig)


    print(f"Run concluída! Acurácia: {acuracia:.2f}")
    print(f"Run f1: {f1}")
    # print(f"Run condusion matrix: {matrix}")
    # print(f"run classification report: {classification}")

# %%
df.head(5)