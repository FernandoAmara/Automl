#Fernando Amaral
import autokeras as ak
import pandas as pd
from sklearn.model_selection import train_test_split

#Importa dados
imp = pd.read_csv('Churn_treino.csv', sep=";")

#Separa variaveis independentes da classe
X = imp.iloc[:,0:10]
y = imp.iloc[:,10]


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)

# Inicializa com 10 modelos diferentes
modelo = ak.StructuredDataClassifier(max_trials=10) 

#Cria o modelo
modelo.fit( x= X_train, y =y_train, epochs=100)

modelo.evaluate(x=X_test, y=y_test)

#Previsão
prever = pd.read_csv('Churn_prever.csv', sep=";")
previsao = modelo.predict(prever)
