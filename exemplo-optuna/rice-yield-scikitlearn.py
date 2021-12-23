import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPRegressor

# importando as métricas que serão utilizadas
# para conferir mais métricas (ou detalhes dessas que importamos)
# acesse: https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score


def mean_bias_error(real, previsto):
    return np.sum(real - previsto) / len(previsto)


def preprocess(df, train_perc = 0.85):
    """
    Recebe um dataframe `df` para ser préprocessado separando
    """

    columns_to_delete = [
        "FLUVENTS",
        "DYSTROPEPTS",
        "ORTHENTS",
        "UDALFS",
        "USTALFS",
    ]

    df.drop(columns = columns_to_delete, inplace = True)

    target = df['RICE-YIELD']
    data   = df.drop(columns = ['RICE-YIELD'])

    xtrain, xtest, ytrain, ytest = train_test_split(data, target,
                                                    train_size = train_perc,
                                                    shuffle = True)

    # scalers de treinamento serão descartados
    data_scaler   = StandardScaler()
    target_scaler = StandardScaler()

    norm_xtrain = data_scaler.fit_transform(xtrain)
    norm_ytrain = target_scaler.fit_transform(ytrain.values.reshape(-1, 1))

    norm_xtest = data_scaler.transform(xtest)
    norm_ytest = target_scaler.transform(ytest.values.reshape(-1, 1))

    return norm_xtrain, norm_ytrain, norm_xtest, norm_ytest, data_scaler, target_scaler


def train(x: "norm_xtrain", y: "norm_ytrain"):
    """
    Treina uma Rede Neural utilizando os conjuntos de treinamento
    `xtrain` e `ytrain` e retorna o modelo treinado.
    """
    modelo = MLPRegressor(verbose = 1,
                          hidden_layer_sizes = (300, 100),
                          solver = 'adam',
                          tol = 1e-6,
                          n_iter_no_change = 20,
                          max_iter = 1000)
    modelo.fit(x, y)
    return modelo


def evaluate(model, x: "norm_xtest", y: "norm_ytest", yscaler):
    """
    Avalia o modelo treinado nos conjuntos de teste `xtest` e `ytest`
    e imprime na tela os valores das métricas obtidos.
    """
    norm_previsoes = model.predict(x) # realizando a previsão

    # transformação (de volta) da escala dos dados
    previsoes     = yscaler.inverse_transform(norm_previsoes.reshape(-1, 1))
    valores_reais = yscaler.inverse_transform(y.reshape(-1, 1))

    # mostranso as métricas obtidas com o modelo no conjunto de teste
    print("MAE =", mean_absolute_error(valores_reais, previsoes))
    print("RMSE =", mean_squared_error(valores_reais, previsoes, squared = False))
    print("R2 =", r2_score(valores_reais, previsoes))
    print("MAPE =", mean_absolute_percentage_error(valores_reais, previsoes))
    print("MBE =", mean_bias_error(valores_reais, previsoes))


if __name__ == "__main__":

    data = pd.read_csv("./X1.csv")
    target = pd.read_csv("./y1.csv")

    data['RICE-YIELD'] = target

    xtrain, ytrain, xtest, ytest, xscaler, yscaler = preprocess(data)

    neural_net = train(xtrain, ytrain)

    evaluate(neural_net,
             x = xtest,
             y = ytest,
             yscaler = yscaler)

