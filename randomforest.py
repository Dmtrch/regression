import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Генерируем данные
X = np.linspace(0, 10, 100)[:, np.newaxis]
y = np.sin(X) + np.random.normal(0, 0.15, size=100)[:, np.newaxis]

# Разбиваем выборки
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

# Создаем модель случайного леса
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=0)

# Обучаем модель
rf_regressor.fit(X_train, y_train.ravel())

# Предсказания на тесте
y_pred = rf_regressor.predict(X_test)

# Визуализация
plt.plot(X_test, y_test, 'b-', label='Actual')
plt.plot(X_test, y_pred, 'r--', label='Predicted')
plt.legend()
plt.show()

# Метрики
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')
