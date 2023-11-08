from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Генерируем больше данных
X = np.linspace(0, 10, 50)[:, np.newaxis]
y = np.sin(X) + np.random.normal(0, 0.3, size=50)[:, np.newaxis]

# Разбиваем на тест и трейн
X_train, X_test = X[:40], X[40:]
y_train, y_test = y[:40], y[40:]

pipeline = make_pipeline(
    StandardScaler(), # Нормализация данных
    PolynomialFeatures(degree=3), # Полином 3-й степени
    Lasso(alpha=0.01) # Регуляризация Lasso
)

# Обучаем модель
pipeline.fit(X_train, y_train)

# Делаем предсказания
y_pred = pipeline.predict(X_test)

# Считаем метрики
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


plt.plot(X_test, y_test, 'b.', label='Actual')
plt.plot(X_test, y_pred, 'r.', label='Predicted')
plt.title('Regression Results')
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.show()

print(f"RMSE: {rmse}")
print(f"R-squared: {r2}")
