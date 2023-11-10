import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


df = pd.read_excel('sales.xlsx')
df.set_index('month',inplace = True)
df.head()


# # Подготовка временного ряда
time_series_data = df[:36]
test = df[36:48]


#
# Создание и обучение модели ARMA
p, d, q = 0, 1, 4  # Устанавливаем порядки AR и MA, а порядок интеграции (d) равен 0
model = sm.tsa.SARIMAX(time_series_data,
                       order=(p, d, q),
                       seasonal_order=(0,1,1,12))
results = model.fit()

# Прогнозирование на будущее
n_forecast = 12  # Количество точек для прогноза
# forecast = results.forecast(steps=n_forecast)

start = len(time_series_data)

end = len(time_series_data) + n_forecast -1

forecast = results.predict(start , end)



# # # Выводим прогнозные значения
print("Прогнозные значения:")
print(forecast)
#
#
# # Сдвигаем серию forecast на одну позицию вниз
forecast_shifted = forecast.shift(periods=1, fill_value=None)

# Визуализируем исходный временной ряд и прогноз
plt.plot(df, color='black', label='Исходный временной ряд')
plt.plot(forecast_shifted, color = 'red', label='Прогноз', linestyle='--')
plt.plot(test, color = 'blue', label='факт')
# plt.plot(df.rolling(window = 12).mean(), label = 'Скользящее среднее за 12 месяцев основной', color = 'orange')
# plt.plot(forecast_shifted.rolling(window = 12).mean(), label = 'Скользящее среднее за 12 месяцев расчетный', color = 'pink')
# plt.plot(test.rolling(window = 12).mean(), label = 'Скользящее среднее за 12 месяцев тест', color = 'green')
plt.title('Прогноз временного ряда')
plt.legend()
plt.grid(True)
plt.show()
