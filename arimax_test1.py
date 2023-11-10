import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Создаем массив значений по оси x от 0 до 60
x_train = np.arange(0, 60, 0.25)

# Вычисляем значения по оси y согласно вашей формуле
y_train = abs(np.sin(x_train/3) * 0.01 * x_train**2)
y1_train = abs(np.sin((x_train/3) + 3) * 0.02 * x_train**2)
y2_train = (y1_train + y_train) / 2

# Создаем датафрейм
df_train = pd.DataFrame({'x': x_train, 'y_train': y_train})
df1_train = pd.DataFrame({'x': x_train, 'y1_train': y1_train})
df2_train = pd.DataFrame({'x': x_train, 'y2_train': y2_train})

# Построение модели ARIMAX
train_end_point = 60
model = sm.tsa.ARIMA(df2_train['y2_train'].iloc[:train_end_point],
                     exog=pd.concat([df_train['y_train'], df1_train['y1_train']], axis=1).iloc[:train_end_point],
                     order=(1, 0, 1))
results = model.fit()

# Получение прогнозов

# Создаем массив значений по оси x от 0 до 100
x = np.arange(60, 100, 0.25)

# Вычисляем значения по оси y согласно вашей формуле
y = abs(np.sin(x/3) * 0.01 * x**2)
y1 = abs(np.sin((x/3) + 3) * 0.02 * x**2)

# Создаем датафрейм
df = pd.DataFrame({'x': x, 'y': y})
df1 = pd.DataFrame({'x': x, 'y1': y1})

forecast_steps = 160

exog_forecast = pd.concat([df['y'], df1['y1']], axis=1).iloc[:forecast_steps]

forecast = results.get_forecast(steps=forecast_steps, exog=exog_forecast)

forecast_ci = forecast.conf_int()

# Заново считаем y2_test до 100
x_test = np.arange(0, 100, 0.25)
y_test = abs(np.sin(x_test/3) * 0.01 * x_test**2)
y1_test = abs(np.sin((x_test/3) + 3) * 0.02 * x_test**2)
y2_test = (y1_test + y_test) / 2
df2_test = pd.DataFrame({'x_test': x_test, 'y2_test': y2_test})

# Обновленное создание forecast_index
forecast_index = df2_test['x_test'][train_end_point:train_end_point + forecast_steps]

# Создаем график
plt.plot(df2_test['x_test'], df2_test['y2_test'], label='Observed', color='green', linewidth=3)
plt.plot(forecast_index+45, forecast.predicted_mean, label='Forecast', color='red', linestyle='--',linewidth=1)
plt.fill_between(forecast_index+45, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='blue', alpha=0.2)
plt.xlabel('x_test')
plt.ylabel('y2_test')
plt.title('ARIMAX Forecast for y2')
plt.legend()
plt.grid(True)

# Отображаем график
plt.show()
