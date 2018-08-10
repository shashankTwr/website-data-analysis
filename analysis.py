import pandas as pd
import numpy as np
from fbprophet import Prophet
data_file = "website-traffic2.txt"
df = pd.read_csv(data_file)
from time import strptime
df['Month'] = df['Month'].apply(lambda x: strptime(x,'%B').tm_mon)
df['dates']=df['Year'].astype(str)+'-'+df['Month'].astype(str)+'-'+df['Day'].astype(str)
print(df['dates'])
df['Day Index'] = pd.to_datetime(df['dates'])
##cleaning extra added column due to us changing initial df in object type to datetime64
df.drop('dates', axis=1, inplace=True)
print(df.dtypes)
print(df.head())
df.drop('Month', axis=1, inplace=True)
df.drop('Year', axis=1, inplace=True)
df.drop('Day', axis=1, inplace=True)
df.drop('DayOfWeek',axis=1,inplace=True)
df.set_index('Day Index').plot()
df['Visits'] = np.log(df['Visits'])
df['y']=df['Visits']
df['ds']=df['Day Index']
df.drop('Visits', axis=1, inplace=True)
df.drop('Day Index',axis=1,inplace=True)

print(df.head())
m1 = Prophet()
m1.fit(df)

future1 = m1.make_future_dataframe(periods=30)
forecast1 = m1.predict(future1)

forecast1[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

np.exp(forecast1[['yhat', 'yhat_lower', 'yhat_upper']].tail())

m1.plot(forecast1)

m1.plot_components(forecast1)
articles = pd.DataFrame({
  'holiday': 'publish',
  'ds': pd.to_datetime(['2009-09-27', '2009-10-05', '2009-10-14', '2009-10-26', '2009-11-9',
                        '2009-11-18', '2009-11-30', '2009-12-17', '2009-12-29']),
  'lower_window': 0,
  'upper_window': 5,
})
print(articles.head())
m2 = Prophet(holidays=articles).fit(df)
future2 = m2.make_future_dataframe(periods=90)
forecast2 = m2.predict(future2)
m2.plot(forecast2)

m2.plot_components(forecast2);


m3 = Prophet(holidays=articles, mcmc_samples=500).fit(df)
future3 = m3.make_future_dataframe(periods=90)
forecast3 = m3.predict(future3)
forecast3["Sessions"] = np.exp(forecast3.yhat).round()
forecast3["Sessions_lower"] = np.exp(forecast3.yhat_lower).round()
forecast3["Sessions_upper"] = np.exp(forecast3.yhat_upper).round()
forecast3[(forecast3.ds > "3-5-2017") &
          (forecast3.ds < "4-1-2017")][["ds", "yhat", "Sessions_lower",
                                        "Sessions", "Sessions_upper"]]