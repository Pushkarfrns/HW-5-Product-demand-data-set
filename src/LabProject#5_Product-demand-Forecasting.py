
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df_prod=pd.read_csv('C:\\Users\\p860n111\\Desktop\\data science\\HW-5-Product-demand-data-set\\Historical Product Demand.csv')


# In[4]:


# so we can see that only Date column has missing values
df_prod.info()


# In[5]:


# There are total 2160 unique products.
# The most popular product is Product_1359 with 16936 frequency.
# Only 4 unqique warehouses
# All in all total 33 unique Product_Category

df_prod.describe()


# In[6]:


# to handle missing attribute, dropping the NA values for date field so, now we have 1037336 entries/records.
df_prod=df_prod.dropna()


# In[9]:


df_prod.info()


# In[10]:


df_prod.shape


# In[11]:


len(df_prod['Product_Code'].unique())


# In[12]:


len(df_prod['Warehouse'].unique())


# In[13]:


# so we have 2160 prodtcs across 33 different categories
len(df_prod['Product_Category'].unique())


# In[14]:


len(df_prod['Date'].unique())


# In[15]:


dates = [pd.to_datetime(date) for date in df_prod['Date']]
dates.sort()


# In[16]:


# date of first order:  8th January 2011
dates[0]


# In[17]:


# date of last  order:  9th January 2017
dates[-1]


# In[19]:


#addtiomnal info1: To find number of products that belong to each category.
df_prod.groupby(['Product_Category' ]).count()['Product_Code']


# In[20]:


df_prod_ProductPerCategory=df_prod.groupby(['Product_Category' ]).count()['Product_Code']
df_prod_ProductPerCategory= df_prod_ProductPerCategory.to_frame()
df_prod_ProductPerCategory.head()


# In[22]:


df_prod_ProductPerCategory=df_prod_ProductPerCategory.sort_values('Product_Code',ascending=False)


# In[23]:


df_prod_ProductPerCategory.head(10)


# In[24]:


df_prod_ProductPerCategory


# In[25]:


df_prod_ProductPerCategory = df_prod_ProductPerCategory.rename(columns={'Product_Code': 'Number of Products'})


# In[26]:


df_prod_ProductPerCategory.head(10)


# In[29]:


from PIL import Image
#Matplot to visualize data, also Seaborn and pandas do this
import matplotlib.pyplot as plt
import seaborn as sns
# Inline to show images in jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(15,15))
ax = sns.barplot(x='Number of Products',y='Product_Category',data=df_prod_ProductPerCategory)
ax.set(xlabel='Category of product', ylabel='Number of Products')
plt.show()


# In[32]:


df_prod_ProductPerCategory['Product_Category'] = df_prod_ProductPerCategory.index.tolist()
df_prod_ProductPerCategory.columns = ['Number of Products','Product_Category']
df_prod_ProductPerCategory.index= np.arange(0,len(df_prod_ProductPerCategory))
df_prod_ProductPerCategory


# In[35]:


# Visualization: category versus number of products belong to each category
from PIL import Image
#Matplot to visualize data, also Seaborn and pandas do this
import matplotlib.pyplot as plt
import seaborn as sns
# Inline to show images in jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(15,15))
ax = sns.barplot(x='Number of Products',y='Product_Category',data=df_prod_ProductPerCategory)
ax.set(xlabel='Category of product', ylabel='Number of Products')
plt.show()


# In[39]:


df_prod_ProductPerCategory.to_csv('AdditionalInfo#1_NoOfProductEachCategory.csv', sep=',')


# In[40]:


#addtiomnal info2: To find number of product category that belong to each warehouse.
df_prod.groupby(['Warehouse' ]).count()['Product_Category']


# In[62]:


df_prod_CategoryPerWHouse=df_prod.groupby(['Warehouse' ]).count()['Product_Category'].unique()
df_prod_CategoryPerWHouse= df_prod_CategoryPerWHouse.to_frame()
df_prod_CategoryPerWHouse.head()


# In[63]:


df_prod_CategoryPerWHouse['Warehouse'] = df_prod_CategoryPerWHouse.index.tolist()
df_prod_CategoryPerWHouse.columns = ['Number of Products','Product_Category']
df_prod_CategoryPerWHouse.index= np.arange(0,len(df_prod_CategoryPerWHouse))
df_prod_CategoryPerWHouse


# In[55]:


df_prod_CategoryPerWHouse.head()


# In[54]:


df_prod_CategoryPerWHouse.drop(['Warehouse'], axis = 1, inplace = True, errors = 'ignore')


# In[64]:


df_prod_CategoryPerWHouse = df_prod_CategoryPerWHouse.rename(columns={'Number of Products': 'Number of Categories'})


# In[65]:


df_prod_CategoryPerWHouse = df_prod_CategoryPerWHouse.rename(columns={'Product_Category': 'Warehouse'})


# In[66]:


df_prod_CategoryPerWHouse.head()


# In[68]:


# Visualization 2: categories versus warehouses
plt.figure(figsize=(15,15))
ax = sns.barplot(x='Number of Categories',y='Warehouse',data=df_prod_CategoryPerWHouse)
ax.set(xlabel='Number of categories', ylabel='Warehouse')
plt.show()


# In[69]:


df_prod.groupby(['Warehouse','Product_Category']).count()['Product_Code']


# In[78]:


df_prod.groupby(['Warehouse' ]).count()['Product_Category']


# In[81]:


#AdditionalInformation3: For each Warehouse to find corresponsing Product_Category and for each product category
#find total product. 
df_prod.groupby(['Warehouse','Product_Category']).count()['Product_Code']


# In[82]:


# Now converting the above data into a frame (playWise_lines_per_player).

df_prod_WHouse_Category_product= df_prod.groupby(['Warehouse','Product_Category']).count()['Product_Code']
df_prod_WHouse_Category_product= df_prod_WHouse_Category_product.to_frame()
df_prod_WHouse_Category_product


# In[86]:


df_prod_WHouse_Category_product
df_prod_WHouse_Category_product.to_csv('AdditionalInfo#3_WHouse_Category_product.csv', sep=',')


# In[85]:


df_prod_WHouse_Category_product


# In[87]:


#addtiomnal info4: To find the date on which maximum and minimum orders where placed.
df_prod.groupby(['Date' ]).count()['Order_Demand']


# In[92]:


df_prod_OrderPerDay=df_prod.groupby(['Date' ]).count()['Order_Demand']
df_prod_OrderPerDay= df_prod_OrderPerDay.to_frame()
df_prod_OrderPerDay=df_prod_OrderPerDay.sort_values('Order_Demand',ascending=False)
df_prod_OrderPerDay.head()


# In[93]:


# so from from above result we can conclude that on 2013/9/27, maximum of 2075 orders were demand.

df_prod_OrderPerDay['Date'] = df_prod_OrderPerDay.index.tolist()
df_prod_OrderPerDay.columns = ['Order_Demand','Date']
df_prod_OrderPerDay.index= np.arange(0,len(df_prod_OrderPerDay))
df_prod_OrderPerDay


# In[98]:


# Visualization 3: Date wise order demand distribution
plt.figure(figsize=(15,500))
ax = sns.barplot(x='Order_Demand',y='Date',data=df_prod_OrderPerDay)
ax.set(xlabel='Number of Demand ', ylabel='date')
plt.show()


# In[99]:


df_prod_OrderPerDay.to_csv('AdditionalInfo#4_Demans_order_per_day.csv', sep=',')


# In[128]:


#Additiomnal info5: to access/get date, month and year information from date column
df_prod_new= df_prod


# In[129]:


df_prod_new.head()


# In[130]:


#created 3 new column for year month and date
df_prod_new['year'] = pd.DatetimeIndex(df_nba_elo_new['Date']).year
df_prod_new['month'] = pd.DatetimeIndex(df_nba_elo_new['Date']).month
df_prod_new['date'] = pd.DatetimeIndex(df_nba_elo_new['Date']).day


# In[131]:


df_prod_new.head()


# In[141]:


#Additiomnal info#6: The year in which maximum number of orders were demanded
df_prod_new_Year=df_prod_new.groupby(['year']).count()['Order_Demand']


# In[142]:


df_prod_new_Year.head()


# In[143]:


df_prod_new_Year= df_prod_new_Year.to_frame()
df_prod_new_Year['year'] = df_prod_new_Year.index.tolist()
df_prod_new_Year.columns = ['Number of product demand in year','year']
df_prod_new_Year.index= np.arange(0,len(df_prod_new_Year))
df_prod_new_Year.head()


# In[148]:


# Visualization : year wise demand plot
plt.figure(figsize=(8,8))
ax = sns.barplot(x='year',y='Number of product demand in year',data=df_prod_new_Year)
ax.set(xlabel='year', ylabel='Number of product demand')
plt.show()


# In[153]:


#Additiomnal info#7: The month in which maximum number of deamnds were there.
df_prod_new_Month=df_prod_new.groupby(['month']).count()['Order_Demand']
df_prod_new_Month.head()
df_prod_new_Month= df_prod_new_Month.to_frame()

#df_prod_new_Month = df_prod_new_Month.rename(columns={'team1': 'Number of Matches Played'})
#df_prod_new_Month = df_prod_new_Month.rename(columns={'team1': 'Number of Matches Played'})
df_prod_new_Month=df_prod_new_Month.sort_values('Order_Demand',ascending=False)
df_prod_new_Month


# In[154]:


df_prod_new_Month['month'] = df_prod_new_Month.index.tolist()
df_prod_new_Month.columns = ['Number of product demand in year','month']
df_prod_new_Month.index= np.arange(0,len(df_prod_new_Month))
df_prod_new_Month.head()


# In[157]:


# Visualization : month wise demand plot
plt.figure(figsize=(15,8))
ax = sns.barplot(x='month',y='Number of product demand in year',data=df_prod_new_Month)
ax.set(xlabel='month', ylabel='Number of product demand')
plt.show()


# In[158]:


#Additiomnal info#8: The date in which maximum number of demands were there.
df_prod_new_date=df_prod_new.groupby(['date']).count()['Order_Demand']
df_prod_new_date.head()
df_prod_new_date= df_prod_new_date.to_frame()

#df_prod_new_Month = df_prod_new_Month.rename(columns={'team1': 'Number of Matches Played'})
#df_prod_new_Month = df_prod_new_Month.rename(columns={'team1': 'Number of Matches Played'})
df_prod_new_date=df_prod_new_date.sort_values('Order_Demand',ascending=False)
df_prod_new_date


# In[160]:


df_prod_new_date['date'] = df_prod_new_date.index.tolist()
df_prod_new_date.columns = ['Number of product demand in year','date']
df_prod_new_date.index= np.arange(0,len(df_prod_new_date))
df_prod_new_date.head()
# Visualization : date wise demand plot
plt.figure(figsize=(15,8))
ax = sns.barplot(x='date',y='Number of product demand in year',data=df_prod_new_date)
ax.set(xlabel='date', ylabel='Number of product demand')
plt.show()

## end of additional information and EDA


# In[162]:


df_prod_new.head()


# In[163]:


df_prod_new.info()


# In[165]:


df_prod_new.dtypes


# In[71]:


import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'


# In[57]:


df_prod


# In[60]:


#There are several warehouses in the dataset, we start from time series analysis and forecasting for Warehouse: Whse_A.
warehouseA.info()


# In[59]:


warehouseA = df_prod.loc[df_prod['Warehouse'] == 'Whse_A']


# In[61]:


warehouseA.shape
from datetime import datetime
warehouseA['Date']=pd.to_datetime(warehouseA['Date'])
warehouseA.head()


# In[64]:


# to find the minimum and maximum date
warehouseA['Date'].min(), warehouseA['Date'].max()
warehouseA.isnull().sum()


# In[63]:


cols = ['Product_Code', 'Warehouse', 'Product_Category']
warehouseA.drop(cols, axis=1, inplace=True)
warehouseA.info()
warehouseA.head()


# In[65]:


# since 911 records have negative value from order_demand so converting the order demand to be of float column from object type
# and then removing the negative value records so 911 removed 
cols = ['Order_Demand']
warehouseA[cols] = warehouseA[cols].apply(lambda x: pd.to_numeric(x.astype(str)
                                                   .str.replace(',',''), errors='coerce'))
# to handle missing attribute, dropping the NA values for date field so, now we have 1037336 entries/records.
warehouseA=warehouseA.dropna()
#warehouseA = warehouseA.sort_values('Date')

#warehouseA.head()


# In[66]:


warehouseA = warehouseA.groupby('Date')['Order_Demand'].sum().reset_index()


# In[67]:


warehouseA.info()
warehouseA=warehouseA.dropna()


# In[68]:


warehouseA = warehouseA.set_index('Date')
warehouseA.index


# In[34]:


from datetime import datetime
warehouseA['Date']=pd.to_datetime(warehouseA['Date'])


# In[35]:


warehouseA.head()


# In[69]:


y = warehouseA['Order_Demand'].resample('MS').mean()


# In[72]:


#Visualizing Furniture Sales Time Series Data
y.plot(figsize=(15, 6))
plt.show()


# In[359]:


#y.dropna()
y.index[y.isnull()]


# In[74]:


y['2016':]


# In[82]:


y=y.dropna()


# In[83]:


y.shape


# In[85]:


y.isnull().sum()


# In[80]:


p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# In[86]:


for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


# In[87]:


mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])


# In[88]:


results.plot_diagnostics(figsize=(16, 8))
plt.show()


# In[121]:


#Validating forecasts

pred = results.get_prediction(start=(52), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2011':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Order Date')
ax.set_ylabel('Warehouse sale/demand')
plt.legend()
plt.show()


# In[135]:


y_forecasted = pred.predicted_mean
y_truth = y['2016-01-03':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))


# In[136]:


print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

