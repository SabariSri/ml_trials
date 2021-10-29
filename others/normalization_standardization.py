from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

data1 = {
    'int': [2.5, 1.5, 0.5, 4.0],
    'amt': [2000, 6000, 1000, 4000],
    'age': [40, 25, 30, 65]
}

df = pd.DataFrame(data1)
print(df.to_string(index=False))

scaler1 = MinMaxScaler()
norm_data = scaler1.fit_transform(df)
print('\n', norm_data)

scaler2 = StandardScaler()
std_data = scaler2.fit_transform(df)
print('\n', std_data)
