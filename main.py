from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch import nn, optim, Tensor
from torch.nn.modules.loss import _Loss, _WeightedLoss
from torch.optim import Optimizer

from models.neural_network import NeuralNet

# https://www.kaggle.com/competitions/titanic/data


def pre_process(df_main: DataFrame) -> DataFrame:
    df = df_main.copy()
    df['Sex'] = df['Sex'].map({'male': 0,
                               'female': 1})
    df['Name'] = df['Name'].str.extract(',\s(Mr\.|Mrs\.|Miss\.)')
    df['Name'] = df['Name'].map({'Mr.': 0,
                                 'Miss.': 1,
                                 'Mrs.': 2,
                                 np.nan: 3})
    df['Ticket'] = df['Ticket'].str.extract('(\d+)$')
    df['Embarked'] = df['Embarked'].map({'Q': 0,
                                         'S': 1,
                                         'C': 2,
                                         np.nan: 3})

    def transform_cabin(cabin):
        if pd.isna(cabin):
            return np.nan
        # Splitting the cabin string to handle cases like 'C56 C67'
        part = cabin.split()[0]
        val = ord(part[0]) - ord('A')
        val = val * 1000
        # The first character is the letter, the rest is the number
        if ' ' in part:
            part = part.split(' ')[0]
        number = val + int(0 if part[1:] == '' else part[1:])
        return number

    df['Cabin'] = df['Cabin'].apply(transform_cabin)
    for column in df.columns:
        has_nans = df[column].isna().any()
        print(f"Column {column} has NaNs: {has_nans}")

    df.fillna(-1, inplace=True)
    df['Ticket'] = df['Ticket'].astype(int)

    return df

def get_min_max_values(df_test: DataFrame, df_train: DataFrame) -> np.ndarray:
    arr: np.ndarray = np.zeros(shape=(len(df_train.columns), 2))
    for column_index, column in enumerate(df_test.columns):
        arr[column_index, 0] = min(df_test[column].min(), df_train[column].min())
        arr[column_index, 1] = max(df_test[column].max(), df_train[column].max())
    return arr

def normalize_data(df: DataFrame, min_max_values: np.ndarray):
    for i, column in enumerate(df.columns):
        min_val, max_val = min_max_values[i]
        df[column] = (df[column] - min_val) / (max_val - min_val)

if __name__ == '__main__':
    folder: str = '/home/jes/Documents/GitRepos/TitanicKaggle/data/'
    df_test = pre_process(pd.read_csv(folder + 'test.csv'))
    df_train = pre_process(pd.read_csv(folder + 'train.csv'))
    df_train_mod = df_train.drop('Survived', axis=1)
    df_train_mod = df_train_mod.drop('PassengerId', axis=1)
    df_test_mod = df_test.drop('PassengerId', axis=1)

    min_max_values: np.ndarray = get_min_max_values(df_test_mod, df_train_mod)
    normalize_data(df_train_mod, min_max_values)

    # y = df_train['Survived'].to_numpy().reshape(-1, 1)
    y = pd.get_dummies(df_train, columns=['Survived'])[['Survived_0', 'Survived_1']].to_numpy().astype(float)
    # one_hot_encoded_df[['Category_A', 'Category_B', 'Category_C']].to_numpy()
    X = df_train_mod.to_numpy()

    X_tensor = torch.tensor(X, dtype=torch.float).to(torch.device('cuda'))
    y_tensor = torch.tensor(y, dtype=torch.float).to(torch.device('cuda'))

    neural_net = NeuralNet(input_features=len(df_train_mod.columns), out_features=2).to(torch.device('cuda'))

    criterion: _WeightedLoss = nn.CrossEntropyLoss()
    optimizer: Optimizer = optim.Adam(neural_net.parameters())

    epochs = 1000
    batch_size = 16

    for epoch in range(epochs):
        for i in range(0, len(X_tensor), batch_size):
            # Get the mini-batch
            X_batch = X_tensor[i:i + batch_size]
            y_batch = y_tensor[i:i + batch_size]

            # RuntimeError: mat1 and mat2 shapes cannot be multiplied (16x11 and 12x256)
            outputs: Tensor = neural_net.forward(X_batch)
            loss: Tensor = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    print('finished training')

    with torch.no_grad():
        normalize_data(df_test_mod, min_max_values)
        test_mod = df_test_mod.to_numpy()
        test_tensor = torch.tensor(test_mod, dtype=torch.float).to(torch.device('cuda'))
        predictions = torch.softmax(neural_net.forward(test_tensor), dim=1)
        predicted_classes = torch.argmax(predictions, dim=1)
    outputs = np.concatenate((df_test['PassengerId'].to_numpy()[:, None], predicted_classes.detach().cpu().numpy()[:, None]), axis=1)
    df = pd.DataFrame(outputs, columns=['PassengerId', 'Survived'])
    df.to_csv('results.csv', index=False)