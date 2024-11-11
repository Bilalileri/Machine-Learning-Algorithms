import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

df= pd.read_csv("C:/Users/Bilal İLERİ/Desktop/Data Science/Exercise/canada_per_capita_income.csv")
print(df.head())

plt.xlabel('Year')
plt.ylabel('Per Capita Income (US$)')
plt.scatter(df['year'], df['per capita income (US$)'], color='red', marker='+')
plt.show()

year = df['year']
price = df['per capita income (US$)']

reg = linear_model.LinearRegression()
reg.fit(df[['year']], df['per capita income (US$)'])

print(reg.predict([[2020]]))

