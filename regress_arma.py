import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt



# Создаем массив значений по оси x от 1 до 100
x = np.arange(1, 100)

# Вычисляем значения по оси y согласно вашей формуле
y = abs((np.sin(x) * ((4*x)/(5*x))))

# Создаем датафрейм
df = pd.DataFrame({'x': x, 'y': y})

# Подготовка временного ряда
time_series_data = df['y']

# Создание и обучение модели ARMA
p, d, q = 7, 1, 4  # Устанавливаем порядки AR и MA, а порядок интеграции (d) равен 0
model = sm.tsa.ARIMA(time_series_data, order=(p, d, q))
results = model.fit()

# Прогнозирование на будущее
n_forecast = 40  # Количество точек для прогноза
forecast = results.forecast(steps=n_forecast)


# # Выводим прогнозные значения
# print("Прогнозные значения:")
# print(forecast)

# Создаем массив значений по оси x от 1 до 100
x = np.arange(100, 140)

# Вычисляем значения по оси y согласно вашей формуле
y = abs((np.sin(x) * ((4*x)/(5*x))))

# Создаем датафрейм
df_1 = pd.DataFrame({'x': x, 'y': y})

df3 = pd.concat([df,df_1], ignore_index= True)

# Сдвигаем серию forecast на одну позицию вниз
forecast_shifted = forecast.shift(periods=1, fill_value=None)

# Визуализируем исходный временной ряд и прогноз
plt.plot(df3['x'], df3['y'], label='Исходный временной ряд')
plt.plot(range(len(time_series_data), len(time_series_data) + n_forecast), forecast_shifted, label='Прогноз', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Прогноз временного ряда')
plt.legend()
plt.grid(True)
plt.show()
