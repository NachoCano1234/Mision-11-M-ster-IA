import os
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def train():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    n_estimators = 60
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    mlflow.set_experiment('mision11_iris')
    with mlflow.start_run():
        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_param('random_state', 42)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.sklearn.log_model(model, artifact_path='model', skops_trusted_types=['sklearn.tree._tree.Tree'])
    Path('models').mkdir(exist_ok=True)
    joblib.dump(model, 'models/model.joblib')
    print(f'accuracy={accuracy:.4f}')
    threshold = float(os.getenv('MIN_ACCURACY', '0.80'))
    if accuracy < threshold:
        raise SystemExit(f'Quality gate FAIL: {accuracy:.4f} < {threshold:.4f}')
    return accuracy

if __name__ == '__main__':
    train()
