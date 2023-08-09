import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump, load


df = pd.read_csv('preprocessed_data.csv')

X = df[['length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight']]
y = df['age']

X_full = df.drop('age', axis=1)
y_class = (y > 12).astype(int)

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_full, y_class, test_size=0.25, random_state=42)

categorical = ['sex']
numeric_features = ['length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight']

ct = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical),
    ('scaling', MinMaxScaler(), numeric_features)
])

# pipeline
pipe = Pipeline([
('transformer', ct),
('model', KNeighborsClassifier())
])

pipe.fit(X_train_full, y_train_full)

pred_pipe = pipe.predict(X_test_full)

params = {'model__n_neighbors' : np.arange(2, 30, 2),
      'model__weights': ['uniform', 'distance'],
      'model__metric': ['manhattan', 'euclidean', 'chebyshev', 'minkowski']}

gs = GridSearchCV(pipe, params, scoring='roc_auc', cv=3, n_jobs=-1, verbose=2)

gs.fit(X_train_full, y_train_full)

dump(gs.best_estimator_, 'trained_model.sav') 
