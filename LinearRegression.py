import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Load dataset
data = pd.read_csv("financial_regression.csv")

# 2. Keep only needed columns and remove NaN rows
data = data[['sp500 open', 'sp500 close']].dropna()

# 3. Input and Output
X = data[['sp500 open']]
y = data['sp500 close']

# 4. Train model
model = LinearRegression()
model.fit(X, y)

# 5. Predict
predicted = model.predict(X)

# 6. Plot
plt.scatter(X, y)
plt.plot(X, predicted)
plt.xlabel("S&P 500 Open")
plt.ylabel("S&P 500 Close")
plt.title("Linear Regression: Open vs Close")
plt.show()

# 7. New prediction
result = model.predict([[120]])
print("Predicted close price for open=120:", result[0])
