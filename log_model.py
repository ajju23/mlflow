from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge

# Importing MLflow library
import mlflow

# set tracking uri
mlflow.set_tracking_uri('/Users/ajay/PycharmProjects/MLflow/mlruns')

# load the diabetes dataset
ds = load_diabetes()

X, y = ds.data, ds.target

# train test and split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=136)

# define parameters
alpha = 0.1
solver = 'svd'

# Tracking model parameters
with mlflow.start_run():


    # Running Random Forest Algorithm
    lr = Ridge(alpha=alpha, solver=solver)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    # log parameters
    mlflow.log_param('alpha', alpha)
    mlflow.log_param('solver', solver)

    # Logging Metrics
    mlflow.log_metric('R2 Score', r2_score(y_test, y_pred))
    mlflow.log_metric('MSE', mean_squared_error(y_test, y_pred))

    # Logging Model
    mlflow.sklearn.save_model(lr,'Regression Model')
