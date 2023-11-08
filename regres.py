import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Создаем массив значений по оси x от 0 до 100
x = np.arange(1, 100)

# Вычисляем значения по оси y согласно вашей формуле
y = abs((np.sin(x) * ((x**2)/(15*x))) /(4 * np.tan(x)))

# Создаем датафрейм
df = pd.DataFrame({'x': x, 'y': y})

df.set_index('x', inplace = True)
df.head()



# Создаем график
plt.plot(df, label='y = sin(x) * x^2')
plt.plot(df.rolling(window = 5).mean(), label = 'Скользящее среднее за 5', color = 'orange')
plt.xlabel('x')
plt.ylabel('y')
plt.title('График функции ')
plt.legend()
plt.grid(True)

# Отображаем график
plt.show()

alpha = 0.2


# первое значение совпадает со значением временного ряда
exp_smoothing = [df['y'].iloc[0]]

# в цикле for последовательно применяем формулу ко всем элементам ряда
for i in range(1, len(df['y'])):
    exp_smoothing.append(alpha * df['y'].iloc[i] + (1 - alpha) * exp_smoothing[i - 1])

# расчитаем следующее значение
print(exp_smoothing[-1])

# добавим кривую сглаживаия в качестве столбца в датафрейм
df['Exp_smoothing'] = exp_smoothing
print(df.tail(3))