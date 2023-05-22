import numpy as np
from sklearn.linear_model import LinearRegression

# Advertising expenditure (independent variable)
advertising = [200, 300, 400, 500, 600]
# Sales or customer behavior (dependent variable)
sales = [4000, 4500, 4800, 5200, 5500]

# Perform linear regression
X = np.array(advertising).reshape(-1, 1)
y = np.array(sales)
model = LinearRegression()
model.fit(X, y)

# Predict the next value of sales/customer behavior
next_advertising = 700  # Assuming a new advertising expenditure
next_X = np.array(next_advertising).reshape(1, -1)
next_y = model.predict(next_X)

print("Predicted next value of sales/customer behavior:", next_y)
