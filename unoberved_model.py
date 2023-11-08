import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Создаем массив значений по оси x от 1 до 100
x = np.arange(1, 100)

# Вычисляем значения по оси y согласно вашей формуле
y = abs((np.sin(x) * ((x**2)/(15*x))) /(4 * np.tan(x)))

# Создаем датафрейм
df = pd.DataFrame({'x': x, 'y': y})

# Подготовка временного ряда
time_series_data = df['y']

# Run the HP filter with lambda = 129600
hp_cycle, hp_trend = sm.tsa.filters.hpfilter(time_series_data, lamb=129600)

# Создаем модель UnobservedComponents
model = sm.tsa.UnobservedComponents(time_series_data, 'lltrend')


res = mod.smooth([1., 0, 1. / 129600])
print(res.summary())

# Обучаем модель
#results = model.fit()

# Прогнозирование на будущее
# n_forecast = 40  # Количество точек для прогноза
# forecast = results.get_forecast(steps=n_forecast)

ucm_trend = pd.Series(res.level.smoothed, index=endog.index)

# Извлекаем прогнозные значения
# forecast_values = forecast.predicted_mean

# Создаем массив значений по оси x от 100 до 139
x = np.arange(100, 140)

# Вычисляем значения по оси y согласно вашей формуле
y = abs((np.sin(x) * ((x**2)/(15*x))) /(4 * np.tan(x)))

# Создаем датафрейм для второй части данных
df_1 = pd.DataFrame({'x': x, 'y': y})

df3 = pd.concat([df, df_1], ignore_index=True)

forecast_shifted = forecast_values.shift(periods=1, fill_value=None)

# Визуализируем исходный временной ряд и прогноз
plt.plot(df3['x'], df3['y'], label='Исходный временной ряд')
plt.plot(range(len(time_series_data), len(time_series_data) + n_forecast), forecast_shifted, label='Прогноз', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Прогноз временного ряда с использованием UnobservedComponents')
plt.legend()
plt.grid(True)
plt.show()
