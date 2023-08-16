import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import roc_curve, roc_auc_score
import keras_tuner
from keras.callbacks import EarlyStopping

# Load data
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
dataset = pd.concat([train, test])

# Data Preprocessing
def preprocess_data(df):
    # Handling duplicates
    df.drop_duplicates(subset=df.columns.difference(['id']), inplace=True)
    df.dropna(inplace=True)
    
    # One-hot encoding of 'Type' column
    type_columns = pd.get_dummies(df['Type'], prefix='Type')
    df = pd.concat([df, type_columns], axis=1)
    df.drop(columns=['Type'], inplace=True)
    
    # Frequency encoding of 'Product ID' column
    df['EncodedProductID'] = df.groupby(by=['Product ID'])['Product ID'].transform('count')
    df.drop(columns=['Product ID'], inplace=True)
    
    return df

train = preprocess_data(train)
test = preprocess_data(test)

# Feature Engineering
def create_features(df):
    df["Temperature ratio"] = df['Process temperature [K]'] / df['Air temperature [K]']
    df['Torque * Rotational speed'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']
    df['Torque * Tool wear'] = df['Torque [Nm]'] * df['Tool wear [min]']
    df['TotalFailures'] = df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].sum(axis=1)
    df.drop(['RNF'], axis=1, inplace=True)
    return df

train = create_features(train)
test = create_features(test)

# Model Definition
class MyHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(n_features,)))
        model.add(tf.keras.layers.Dense(units=hp.Int("layer1", min_value=8, max_value=512, step=16), activation='relu'))
        
        if hp.Boolean("BatchNormalization1", default=True):
            model.add(tf.keras.layers.BatchNormalization())
        for i in range(hp.Int('number-of-hidden-layers', 1, 3, default=1)):
            model.add(tf.keras.layers.Dense(units=hp.Int("hidden-layer"+str(i), min_value=32, max_value=512, step=32), activation='relu'))
            if hp.Boolean("BatchNormalization"+str(i), default=True):
                model.add(tf.keras.layers.BatchNormalization())
            if i > 1:
                if hp.Boolean("Dropout"+str(i-1), default=False):
                    model.add(tf.keras.layers.Dropout(hp.Float('dropout-'+str(i-1), 0, 0.5, step=0.1, default=0.5)))
        model.add(tf.keras.layers.Dense(units=hp.Int("final-layer", min_value=32, max_value=512, step=32), activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                      metrics=['AUC'])
        model.summary()
        return model

# Load and preprocess data
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
train = preprocess_data(train)
test = preprocess_data(test)

# Prepare features and target
train_X = train.drop(columns=['id', 'Machine failure']).reset_index(drop=True)
train_y = train['Machine failure'].reset_index(drop=True)
n_features = len(train_X.columns)
test_X = test.drop(columns=['id']).reset_index(drop=True)

# StandardScaler
sc = StandardScaler()
train_X = sc.fit_transform(train_X)
test_X = sc.transform(test_X)

# Splitting data
train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.25)

# Weighting the unbalanced target
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_y), y=train_y)
class_weights = dict(enumerate(class_weights))

# Initialize keras tuner
tuner = keras_tuner.BayesianOptimization(
    hypermodel=MyHyperModel(),
    objective=keras_tuner.Objective("val_auc", direction="max"),
    max_trials=45,
    num_initial_points = 5,
    overwrite=True,
    directory="../hyperOptModelsHistory",
    project_name="BinaryClassificationofMachineFailure",
)

# Search for best hyperparameters
tuner.search(train_X, train_y, epochs=1000, validation_data=(val_X, val_y), class_weight=class_weights, callbacks=[EarlyStopping(patience=30)])

# Get the top 2 models.
models = tuner.get_best_models(num_models=2)
best_model = models[0]

# Build the model.
best_model.build(input_shape=(None, n_features))
best_model.summary()

# Retrain the best model with the best hps
hypermodel = MyHyperModel()
best_hp = tuner.get_best_hyperparameters()[0]
model = hypermodel.build(best_hp)

# Fit with the entire dataset.
train_X = np.concatenate((train_X, val_X))
train_y = np.concatenate((train_y, val_y))
history = hypermodel.fit(best_hp, model, train_X, train_y, epochs=1000, class_weight=class_weights, callbacks=[EarlyStopping(patience=30)], validation_split=0.25)

# Plotting loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.show()

# Generate Prediction
y_pred = model.predict(val_X)
fpr, tpr, _ = roc_curve(val_y, y_pred)
auc = roc_auc_score(val_y, y_pred)
plt.plot(fpr, tpr, label="auc="+str(auc))
plt.legend(loc=4)
plt.show()

print(f"AUC: {auc}")
print(f"\nPrediction:\n {y_pred}")

# Create output
output = pd.DataFrame({"Machine failure": np.squeeze(y_pred)})
print(output[output['Machine failure'] > 0.5].sort_values(by=['Machine failure'], ascending=False))
