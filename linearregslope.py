import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Manually enter the data
prices = [100, 120, 150, 130, 160, 180, 200]
eps = [2.5, 3.0, 3.5, 3.2, 4.0, 4.2, 4.5]

# Perform linear regression
X = np.array(eps).reshape(-1, 1)
y = np.array(prices)
model = LinearRegression()
model.fit(X, y)

# Get the slope and intercept
slope = model.coef_[0]
intercept = model.intercept_

# Plot the regression line
plt.scatter(eps, prices, color='b', label='Data')
plt.plot(eps, model.predict(X), color='r', label='Regression Line')
plt.xlabel('EPS')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Print the slope and intercept
print("Slope:", slope)
print("Intercept:", intercept)
