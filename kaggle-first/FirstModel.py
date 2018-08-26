import pandas as pd
from sklearn.tree import DecisionTreeRegressor

melbourne_file_path = 'melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
# print(melbourne_data.columns)
melbourne_data = melbourne_data.dropna(axis=0)
y=melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
#print(X.head())
melbourne_model = DecisionTreeRegressor(random_state=1)
melbourne_model.fit(X, y)
#print(melbourne_model)
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))