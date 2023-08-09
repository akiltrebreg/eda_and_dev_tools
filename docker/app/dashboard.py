import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load
from explainerdashboard import ClassifierExplainer, ExplainerDashboard

gs = load('trained_model.sav')

df = pd.read_csv('preprocessed_data.csv')

X = df[['length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight']]
y = df['age']

X_full = df.drop('age', axis=1)
y_class = (y > 12).astype(int)

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_full, y_class, test_size=0.25, random_state=42)

# строим дашборд
explainer = ClassifierExplainer(gs, X_test_full.iloc[:10], y_test_full.iloc[:10])
db = ExplainerDashboard(explainer)

db.to_yaml("dashboard.yaml", explainerfile="explainer.dill", dump_explainer=True)
