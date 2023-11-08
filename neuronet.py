import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Генерируем данные
X = np.linspace(0, 10, 100)[:, np.newaxis]
y = np.sin(X) + np.random.normal(0, 0.15, size=100)[:, np.newaxis]

# Разбиваем выборки
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]



# Создаем pipeline
pipe = make_pipeline(
    StandardScaler(),
    MLPRegressor(hidden_layer_sizes=(100,), activation='relu', max_iter=20000)
)

# Обучаем модель
pipe.fit(X_train, y_train.ravel())

# Предсказания на тесте
y_pred = pipe.predict(X_test)


# Визуализация
plt.plot(X_test, y_test, 'b-', label='Actual')
plt.plot(X_test, y_pred, 'r--', label='Predicted')
plt.legend()
plt.show()

# Метрики
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')