import os
import numpy as np
import pandas as pd

import optuna
from optuna.samplers import TPESampler
from optuna.samplers import NSGAIISampler

from functools import partial

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPRegressor

# importando as métricas que serão utilizadas
# para conferir mais métricas (ou detalhes dessas que importamos)
# acesse: https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error


import matplotlib.pyplot as plt


def mean_bias_error(real, previsto):
    return np.sum(real - previsto) / len(previsto)


def preprocess(df, train_perc = 0.85):
    """
    Recebe um dataframe `df` para ser preprocessado separando
    os conjuntos em xtest, ytest e xtrain, ytrain
    """

    # criando um filtro que seleciona todos os atributos apenas 1 valor único
    # fazendo a selação dos atributos invertendo o filtro (o que era True fica False)
    all_equal_columns = df.nunique() == 1
    df = df.loc[:, ~all_equal_columns]

    # lista de colunas que devem ser normalizadas
    normalize_cols = (df.drop(columns = 'RICE-YIELD').nunique() > 20).index.to_list()

    target = df['RICE-YIELD']
    data   = df.drop(columns = ['RICE-YIELD'])

    xtrain, xtest, ytrain, ytest = train_test_split(data, target,
                                                    train_size = train_perc,
                                                    shuffle = True)

    # transformador para normalizar apenas as colunas
    # que devem ser normalizadas
    columns_transformer = ColumnTransformer([
        ("normalizacao", StandardScaler(), normalize_cols)
    ])

    # scalers de treinamento serão descartados
    data_scaler   = columns_transformer
    target_scaler = StandardScaler()

    norm_xtrain = data_scaler.fit_transform(xtrain)
    norm_ytrain = target_scaler.fit_transform(ytrain.values.reshape(-1, 1))

    norm_xtest = data_scaler.transform(xtest)
    norm_ytest = target_scaler.transform(ytest.values.reshape(-1, 1))

    return norm_xtrain, norm_ytrain, norm_xtest, norm_ytest, data_scaler, target_scaler


def objective(trial, norm_x, norm_y, target_scaler):
    """
    Define o objetivo para a otimização feita pelo Optuna.
    """

    xtrain, xvalid, ytrain, yvalid = train_test_split(norm_x, norm_y,
                                                      train_size = 0.85,
                                                      shuffle = True)

    trial.set_user_attr('tol', 1e-6)
    trial.set_user_attr('solver', 'adam')
    trial.set_user_attr('batch_size', 32)
    trial.set_user_attr('early_stopping', True)
    trial.set_user_attr('max_iter', 2000)
    trial.set_user_attr('hidden_layer_sizes', (110, 15))

    param_grid = {
        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-6, 0.01),
        'alpha': trial.suggest_float('alpha', 1e-5, 0.1),
        'beta_1': trial.suggest_float('beta_1', 0.7, 0.9),
        'beta_2': trial.suggest_float('beta_2', 0.7, 0.999),

        # hiperparametros estáticos (não serão otimizados)
        'tol': trial.user_attrs['tol'],
        'solver': trial.user_attrs['solver'],
        'max_iter': trial.user_attrs['max_iter'],
        'batch_size': trial.user_attrs['batch_size'],
        'early_stopping': trial.user_attrs['early_stopping'],
        'hidden_layer_sizes': trial.user_attrs['hidden_layer_sizes']
    }

    model = MLPRegressor().set_params(**param_grid)
    model.fit(xtrain, ytrain)

    pred = target_scaler.inverse_transform( model.predict(xvalid).reshape(-1, 1) )
    real = target_scaler.inverse_transform( yvalid.reshape(-1, 1) )

    return mean_squared_error(real, pred, squared = False)


def train(x: "xtrain", y: "ytrain"):
    """
    Treina uma Rede Neural utilizando os conjuntos de treinamento
    `xtrain` e `ytrain` e retorna o modelo treinado.
    """
    modelo = MLPRegressor(verbose = 1,
                          hidden_layer_sizes = (100, 100),
                          solver = 'adam',
                          tol = 1e-6,
                          n_iter_no_change = 15,
                          max_iter = 1000)
    modelo.fit(x, y)
    return modelo


def evaluate(model, x: "xtest", y: "ytest", yscaler: "scaler para reconstrução"):
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
    print("RMSE = ", mean_squared_error(valores_reais, previsoes, squared = False))
    print("MBE = ", mean_bias_error(valores_reais, previsoes))
    print("MAPE =", mean_absolute_percentage_error(valores_reais, previsoes))


if __name__ == "__main__":

    data = pd.read_csv("./X1.csv")
    target = pd.read_csv("./y1.csv")

    data['RICE-YIELD'] = target

    xtrain, ytrain, xtest, ytest, xscaler, yscaler = preprocess(data)

    nn_study = optuna.create_study(direction = 'minimize',
                                   sampler = NSGAIISampler())

    nn_study.optimize(
        func = partial(objective, norm_x = xtrain, norm_y = ytrain, target_scaler = yscaler),
        timeout = 100, # tempo durante o qual o algoritmo deve ficar "procurando" a melhor arquitetura
        #n_trials = 1, # número de tentativas feitas pelo algoritmo
    )


    # treinando com os melhores parametros encontrados
    rede_neural = MLPRegressor().set_params(**nn_study.best_params)
    rede_neural.fit(xtrain, ytrain)

    evaluate(model = rede_neural,
             x = xtest,
             y = ytest,
             yscaler = yscaler)

    pred = yscaler.inverse_transform( rede_neural.predict(xtest).reshape(-1, 1) )
    real = yscaler.inverse_transform( ytest.reshape(-1, 1) )

    fig, axs = plt.subplots(figsize = (10, 10), dpi = 120)

    axs.scatter(real, pred, alpha = 0.7)
    axs.set_ylabel("Valor esperado")
    axs.set_xlabel("Valor previsto")
    axs.set_aspect("equal", "box")
    plt.show()
