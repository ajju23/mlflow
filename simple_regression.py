from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

# load the diabetes dataset
ds = load_diabetes()

X, y = ds.data, ds.target

# train test and split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=136)

# lets train our dataset
from sklearn.ensemble import RandomForestClassifier

# Running Random Forest Algorithm
rf_clf = LinearRegression()
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)

print('R2 Score: ', r2_score(y_test, y_pred))
print('MSE: ', mean_squared_error(y_test, y_pred))
