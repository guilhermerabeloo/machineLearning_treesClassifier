import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import ExtraTreesClassifier

df = pd.read_csv('wine_dataset.csv')

df['style'] = df['style'].replace('red', 0)
df['style'] = df['style'].replace('white', 1)


# Criando copia da coluna style e criando copia da base de dados sem a coluna style
x = df.drop('style', axis = 1)
y = df['style']

# Separando dados para teste e para treino
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

# Treinando o modelo
model = ExtraTreesClassifier()
model.fit(x_treino, y_treino)

# Usando a base de teste para testar o modelo e verificar a acurácia
result = model.score(x_teste, y_teste)
print(f'Acurácia: {result}')

# Testando com amostragem
print(y_teste[400:403]) # gabarito da coluna 'style' na amostragem de teste nos indices 400 a 403 

predict = model.predict(x_teste[400:403]) # testando quais sao os resultados da coluna 'style' na amostragem de teste nos indices 400 a 403. Deve ser igual ao y_teste
print(predict)

print(df.loc[4708]['style']) # exibindo a linha original de um dos indices acima para conferir