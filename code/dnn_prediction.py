
#import libraries
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt

def create_features(df):
    
    # Create a new feature by subtracting 'Air temperature' from 'Process temperature'
    # df['Temperature difference [K]'] = df['Process temperature [K]'] - df['Air temperature [K]']
    
    # Create a new feature by divided 'Air temperature' from 'Process temperature'
    df["Temperature ratio"] = df['Process temperature [K]'] / df['Air temperature [K]']
    
    # Create a new feature by multiplying 'Torque' and 'Rotational speed' (POWER)
    df['Torque * Rotational speed'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']

    # Create a new feature by multiplying 'Torque' by 'Tool wear'
    df['Torque * Tool wear'] = df['Torque [Nm]'] * df['Tool wear [min]']

    # Create a new feature by adding 'Air temperature' and 'Process temperature'
    # df['Temperature sum [K]'] = df['Air temperature [K]'] + df['Process temperature [K]']
    
    # Create a new feature by multiplying 'Torque' by 'Rotational speed'
    df['Torque * Rotational speed'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']

    df['TotalFailures'] = df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].sum(axis=1)

    df.drop(['RNF'], axis =1, inplace = True)
    
    return df

#import data
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
dataset = pd.concat([train,test])
dataset.head()

#DUPLICATES
train.drop_duplicates(subset=train.columns.difference(['id']),inplace=True)
test.drop_duplicates(subset=test.columns.difference(['id']), inplace=True)
#MISSING VALUES
train.dropna(inplace=True)
test.dropna(inplace=True)
#Searching for non-ordinal categorical features
categorical_columns = train.select_dtypes(include=['object']).columns.values
#Calculating unique values of categorical features
for col in categorical_columns:
    print(f" train {col}.unique = {len(train[col].unique())}, test {col}.unique = {len(test[col].unique())}") 
#ONE-HOT ENCODING of Type column
for df in [train, test]:
    for value in df.Type.unique():
        df[f'Type{value}'] = 0
        df.loc[df.Type == f'{value}', f'Type{value}'] = 1
    df.drop(columns=['Type'], inplace=True)
#Frequency ENCODING of Product ID column (It is a way to utilize the frequency of the categories as labels)
for df in [train, test]:
    df['EncodedProductID'] = df.groupby(by=['Product ID'])['Product ID'].transform('count')
    df.drop(columns=['Product ID'], inplace=True)

train = create_features(train)

train_X = train.drop(columns=['id', 'Machine failure']).reset_index(drop=True)
train_y = train['Machine failure'].reset_index(drop=True)
n_features = len(train_X.columns)

# StandardScaler
sc = StandardScaler() # MinMaxScaler or StandardScaler
train_X = sc.fit_transform(train_X)

model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(n_features,)))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])
model.summary()

#weighting the unbalanced target
#class_weights = dict(enumerate(class_weight.compute_class_weight('balanced', np.unique(train_y), train_y)))

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=100)
print("\n")
print(tf.config.list_physical_devices('GPU'))
print("\n")
#history = model.fit(train_X, train_y, batch_size=256, epochs=1000, class_weight=class_weights, callbacks=[early_stopping], validation_split=0.25)
history = model.fit(train_X, train_y, batch_size=256, epochs=1000, callbacks=[early_stopping], validation_split=0.25)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.show()
