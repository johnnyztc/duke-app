#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import libraries
import pandas as pd
import requests
import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error
from datetime import date
from datetime import datetime
import matplotlib
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
import seaborn as sns
import pytz
import csv
from pathlib import Path
import statsmodels as sm                 
from statsmodels.tools.eval_measures import rmse
import warnings
from datetime import datetime
from datetime import timedelta
import xgboost as xgb
import pytz
from datetime import datetime
import csv
from datetime import timedelta
import plotly.express as px
warnings.filterwarnings("ignore")


# In[2]:


import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px


# In[3]:


TEMPLATE = 'plotly_white'


# In[4]:


# load the latest model we trained
reg_new = xgb.XGBRegressor()
reg_new.load_model('model1.json')


# In[5]:


# Read the raw duk csv
duk = pd.read_csv('duk_model_raw.csv', parse_dates=['Datetime'], index_col=['Datetime'])
duk = duk.drop(columns=duk.columns[0])


# In[6]:


def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df


# In[7]:


def add_lags(duk):
    #target_map = duk['DUK_MW'].to_dict()
    #duk['duk_1_hrs_lag'] = duk['DUK_MW'].shift(1)
    #duk['duk_6_hrs_lag'] = duk['DUK_MW'].shift(6)
    #duk['duk_12_hrs_lag'] = duk['DUK_MW'].shift(12)
    duk['duk_24_hrs_lag'] = duk['DUK_MW'].shift(24) 
    #duk['duk_168_hrs_lag'] = duk['DUK_MW'].shift(168)
    
    #duk['duk_6_hrs_mean'] = duk['DUK_MW'].rolling(window = 6).mean()
    duk['duk_12_hrs_mean'] = duk['DUK_MW'].rolling(window = 12).mean().shift(24)
    duk['duk_24_hrs_mean'] = duk['DUK_MW'].rolling(window = 24).mean().shift(24)  
    
    #duk['duk_6_hrs_std'] = duk['DUK_MW'].rolling(window = 6).std()  
    duk['duk_12_hrs_std'] = duk['DUK_MW'].rolling(window = 12).std().shift(24)
    duk['duk_24_hrs_std'] = duk['DUK_MW'].rolling(window = 24).std().shift(24)
    
    #duk['duk_6_hrs_max'] = duk['DUK_MW'].rolling(window = 6).max()
    duk['duk_12_hrs_max'] = duk['DUK_MW'].rolling(window = 12).max().shift(24)
    duk['duk_24_hrs_max'] = duk['DUK_MW'].rolling(window = 24).max().shift(24)
    duk['duk_168_hrs_max'] = duk['DUK_MW'].rolling(window = 168).max().shift(24)
    
    #duk['duk_6_hrs_min'] = duk['DUK_MW'].rolling(window = 6).min()
    #duk['duk_12_hrs_min'] = duk['DUK_MW'].rolling(window = 12).min()
    #duk['duk_24_hrs_min'] = duk['DUK_MW'].rolling(window = 24).min()
    return duk


# In[8]:


# define t1 to be 24 hours later than current time
t1 = str(date.today() + timedelta(hours=24))


# In[9]:


# load the latest demand data up to today from EIA
def update_demand():
    t0 = str(date.today() + timedelta(hours=24))
    api_key = 'phRLIs3z4GNYWKHtB5d2zunfICyqTsUSpnRJvq2S'
    url = 'https://api.eia.gov/v2/electricity/rto/region-data/data/?frequency=hourly&data[0]=value&facets[respondent][]=DUK&start=2023-02-10T00&end='+t0+'&sort[0][column]=period&sort[0][direction]=asc&offset=0&length=5000&api_key=phRLIs3z4GNYWKHtB5d2zunfICyqTsUSpnRJvq2S'
    r = requests.get(url)
    d = r.json()
    df = pd.json_normalize(d, record_path=['response', 'data'])
    df1 = df[df['type-name'] == 'Demand']
    df1 = df1.reset_index()
    df1.drop('index',axis=1,inplace = True)
    df1['Datetime'] = df1['period']

    for i in range(len(df1)):
        dt=datetime.strptime(df1['period'][i], "%Y-%m-%dT%H")
        df1['Datetime'][i] = dt - timedelta(hours=5)
        i = i+1
    
    df1 = df1.set_index('Datetime')
    df1 = df1[['value']]
    df1 = df1.rename(columns={'value':'DUK_MW'})
    return df1


# In[10]:


df1 = update_demand()
update_demand()


# In[11]:


# load the latest EIA predicted demand data up to today from EIA
def update_prediction_EIA():
    t0 = str(date.today() + timedelta(hours=24))
    api_key = 'phRLIs3z4GNYWKHtB5d2zunfICyqTsUSpnRJvq2S'
    url = 'https://api.eia.gov/v2/electricity/rto/region-data/data/?frequency=hourly&data[0]=value&facets[respondent][]=DUK&start=2023-02-10T00&end='+t0+'&sort[0][column]=period&sort[0][direction]=asc&offset=0&length=5000&api_key=phRLIs3z4GNYWKHtB5d2zunfICyqTsUSpnRJvq2S'
    r = requests.get(url)
    d = r.json()
    df = pd.json_normalize(d, record_path=['response', 'data'])
    df2 = df[df['type-name'] == 'Day-ahead demand forecast']
    df2 = df2.reset_index()
    df2.drop('index',axis=1,inplace = True)
    df2['Datetime'] = df2['period']

    for i in range(len(df2)):
        dt=datetime.strptime(df2['period'][i], "%Y-%m-%dT%H")
        df2['Datetime'][i] = dt - timedelta(hours=5)
        i = i+1
    
    df2 = df2.set_index('Datetime')
    df2 = df2[['value']]
    df2 = df2.rename(columns={'value':'prediction_EIA'})
    return df2


# In[12]:


df2 = update_prediction_EIA()
update_prediction_EIA()


# In[13]:


# determine the prediction range based on the latest available actual demand from EIA
future = pd.date_range('2023-02-09 19:00:00',df1.index[-1] + timedelta(hours=24),freq = '1h')
future_df = pd.DataFrame(index=future)
duk_and_future = pd.concat([duk,future_df])
duk_and_future


# In[14]:


#function to retreive historical and forecast weather data from Visual Crossing
def get_weather(loc, start, end):
    #main URL
    BaseURL = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/'
    
    #user API key
    ApiKey='D5FAK4DB3LVUJGWCTFZRXA2T6'
    
    #location, concatenate if two-word city 
    if len(loc.split()) > 1:
        loc = loc.replace(" ", "")
    Location = loc
    
    #start date, end date inputs
    StartDate = start
    EndDate= end
    
    #choosing csv instead of json
    ContentType="csv"
    
    #selecting hourly data instead of daily
    Include="hours"
    
    #US system instead of metric
    UnitGroup='us'
    
    #incorporating location with API query
    ApiQuery=BaseURL + Location
    
     #accounting for start/end dates selected
    if (len(StartDate)):
        ApiQuery+="/"+StartDate
        if (len(EndDate)):
            ApiQuery+="/"+EndDate

    #adding '?' at end of API query
    ApiQuery+="?"

    #accounting for units, csv or json, and type of information 
    if (len(UnitGroup)):
        ApiQuery+="&unitGroup="+UnitGroup

    if (len(ContentType)):
        ApiQuery+="&contentType="+ContentType

    if (len(Include)):
        ApiQuery+="&include="+Include

    #adding user API key to query
    ApiQuery+="&key="+ApiKey
    
    #change CSV to dataframe
    df0=pd.read_csv(ApiQuery)

    #switch 'datetime' column to 1st column
    df0 = df0[['datetime', 'name', 'temp', 'humidity', 'precip', 'precipprob', 'preciptype','snow','snowdepth', 'windgust', 'windspeed','winddir', 'sealevelpressure', 'cloudcover', 'visibility', 'visibility', 'solarradiation', 'solarenergy', 'uvindex', 'severerisk', 'conditions', 'icon'  ]]

    #remove 'T' that is in every 'datetime' row
    df0['datetime'] = df0['datetime'].map(lambda x: x.replace('T',' '))

    #ensure that 'datetime' column matches EIA API pull for merge
    df0.rename(columns={'datetime': 'Datetime'}, inplace=True)
    
    #set 'datetime' column as datetime object and set as dataframe index
    df0['Datetime'] = pd.to_datetime(df0['Datetime'])
    df0.set_index('Datetime', drop=True, inplace=True)
    df0.index = pd.to_datetime(df0.index)
    
     #remove unwanted weather variables
    df0.drop('preciptype', axis=1, inplace=True) 
    df0.drop('conditions', axis=1, inplace=True) 
    #df0.drop('solarenergy', axis=1, inplace=True)
    df0.drop('icon', axis=1, inplace=True) 
    df0.drop('visibility', axis=1, inplace=True)
    df0.drop('snowdepth', axis=1, inplace=True)
   
    
    #rename columns to include dataframe's city
    #df0 = df0.rename(columns={'temp': loc + '_temp', 'humidity': loc + '_humidity', 'precip': loc + '_precip', 'precipprob': loc + '_precipprob','snow': loc + '_snow', 
                              #'windgust': loc + '_windgust', 'windspeed': loc + '_windspeed', 'winddir': loc + '_winddir',
                              #'sealevelpressure': loc + '_sealevelpressure', 'cloudcover': loc + '_cloudcover',
                              #'solarradiation': loc + '_solarradiation', 'uvindex': loc + '_uvindex',  'severerisk': loc + '_severerisk'})
    
    
    return df0

#function to call API multiple times (look further back) & combine dataframes
#NEED TO MAKE DATES DYNAMIC

def full_df(city):
    first_df = get_weather(city,'2023-2-9',t1)
    return first_df

# Choose 11 counties with DUK Carolinas as a main provider and with biggest population in DUK Carolinas service territory
test_list_3 = ['Mecklenburg County, NC', 'Guilford County, NC', 'Greenville County, SC', 'Forsyth County, NC', 'Spartanburg County, SC', 'Durham County, NC', 'York County, SC', 'Indian Trail, NC', 'Gaston County, NC',
              'Cabarrus County, NC', 'Anderson County, SC']

#function to create dictionary of weather dataframes whose names and corresponding weather are a given list of cities
def create_dfs(names):
    dfs = {}
    for x in names:
        dfs[x] = full_df(x)
    return dfs

#THIS CELL TAKES LONG TIME TO RUN
#create dictionary of weather data dataframes 
dfs_1 = create_dfs(test_list_3) 

#create one dataframe of historical weather for counties in Duke Energy Carolinas coverage area
weatherdf = pd.concat(dfs_1,axis=1)

weatherdf.columns = weatherdf.columns.droplevel(0)    

weatherdf['avg_temp'] = weatherdf[['temp', 'temp', 'temp', 'temp','temp', 'temp','temp', 'temp','temp', 'temp','temp' ]].mean(axis=1)
weatherdf['avg_humidity'] = weatherdf[['humidity', 'humidity', 'humidity', 'humidity','humidity', 'humidity','humidity', 'humidity','humidity', 'humidity','humidity' ]].mean(axis=1)
weatherdf['avg_precip'] = weatherdf[['precip', 'precip', 'precip', 'precip','precip', 'precip','precip', 'precip','precip', 'precip','precip' ]].mean(axis=1)
weatherdf['avg_precipprob'] = weatherdf[['precipprob', 'precipprob', 'precipprob', 'precipprob','precipprob', 'precipprob','precipprob', 'precipprob','precipprob', 'precipprob','precipprob' ]].mean(axis=1)
weatherdf['avg_snow'] = weatherdf[['snow', 'snow', 'snow', 'snow','snow', 'snow','snow', 'snow','snow', 'snow','snow' ]].mean(axis=1)
weatherdf['avg_windgust'] = weatherdf[['windgust', 'windgust', 'windgust', 'windgust','windgust', 'windgust','windgust', 'windgust','windgust', 'windgust','windgust' ]].mean(axis=1)
weatherdf['avg_windspeed'] = weatherdf[['windspeed', 'windspeed', 'windspeed', 'windspeed','windspeed','windspeed','windspeed', 'windspeed','windspeed', 'windspeed','windspeed' ]].mean(axis=1)
weatherdf['avg_winddir'] = weatherdf[['winddir', 'winddir', 'winddir', 'winddir','winddir', 'winddir','winddir', 'winddir','winddir', 'winddir','winddir' ]].mean(axis=1)
weatherdf['avg_sealevelpressure'] = weatherdf[['sealevelpressure', 'sealevelpressure', 'sealevelpressure', 'sealevelpressure','sealevelpressure', 'sealevelpressure','sealevelpressure', 'sealevelpressure','sealevelpressure', 'sealevelpressure','sealevelpressure' ]].mean(axis=1)
weatherdf['avg_cloudcover'] = weatherdf[['cloudcover', 'cloudcover', 'cloudcover', 'cloudcover','cloudcover', 'cloudcover','cloudcover', 'cloudcover','cloudcover', 'cloudcover','cloudcover' ]].mean(axis=1)
weatherdf['avg_solarradiation'] = weatherdf[['solarradiation', 'solarradiation', 'solarradiation', 'solarradiation','solarradiation', 'solarradiation','solarradiation', 'solarradiation','solarradiation', 'solarradiation','solarradiation' ]].mean(axis=1)
weatherdf['avg_solarenergy'] = weatherdf[['solarenergy', 'solarenergy', 'solarenergy', 'solarenergy','solarenergy', 'solarenergy','solarenergy', 'solarenergy','solarenergy', 'solarenergy','solarenergy' ]].mean(axis=1)
weatherdf['avg_uvindex'] = weatherdf[['uvindex', 'uvindex', 'uvindex', 'uvindex','uvindex', 'uvindex','uvindex', 'uvindex','uvindex', 'uvindex','uvindex' ]].mean(axis=1)
weatherdf['avg_severerisk'] = weatherdf[['severerisk', 'severerisk', 'severerisk', 'severerisk','severerisk', 'severerisk','severerisk', 'severerisk','severerisk', 'severerisk','severerisk' ]].mean(axis=1)
weatherdf.head(2)

# Drop column by index using DataFrame.iloc[] and drop() methods.
avg_wdf = weatherdf.drop(weatherdf.iloc[:, 1:165],axis = 1)
avg_wdf


# In[15]:


# prepare the prediction range with the forecasted weather data
avg_wdf = avg_wdf[(avg_wdf.index <= (df1.index[-1] + timedelta(hours=24))) & (avg_wdf.index >= '2023-02-09 19:00:00')]
avg_wdf


# In[16]:


# update the latest demand data into duk_and_future dataframe
for i in df1.index:
    duk_and_future.loc[i,'DUK_MW'] = df1.loc[i].DUK_MW


# In[17]:


# update the latest wewather data into duk_and_future dataframe
for i in avg_wdf.index:
    duk_and_future.loc[i,'avg_temp'] = avg_wdf.loc[i].avg_temp
    duk_and_future.loc[i,'avg_humidity'] = avg_wdf.loc[i].avg_humidity
    duk_and_future.loc[i,'avg_precip'] = avg_wdf.loc[i].avg_precip
    duk_and_future.loc[i,'avg_precipprob'] = avg_wdf.loc[i].avg_precipprob
    duk_and_future.loc[i,'avg_snow'] = avg_wdf.loc[i].avg_snow
    duk_and_future.loc[i,'avg_windgust'] = avg_wdf.loc[i].avg_windgust
    duk_and_future.loc[i,'avg_windspeed'] = avg_wdf.loc[i].avg_windspeed
    duk_and_future.loc[i,'avg_winddir'] = avg_wdf.loc[i].avg_winddir
    duk_and_future.loc[i,'avg_sealevelpressure'] = avg_wdf.loc[i].avg_sealevelpressure
    duk_and_future.loc[i,'avg_cloudcover'] = avg_wdf.loc[i].avg_cloudcover
    duk_and_future.loc[i,'avg_solarradiation'] = avg_wdf.loc[i].avg_solarradiation
    duk_and_future.loc[i,'avg_solarenergy'] = avg_wdf.loc[i].avg_solarenergy
    duk_and_future.loc[i,'avg_uvindex'] = avg_wdf.loc[i].avg_uvindex


# In[18]:


# get the dataframe ready for prediction by adding extra features
duk_and_future = create_features(duk_and_future)
duk_and_future = add_lags(duk_and_future)
duk_and_future['weekofyear']=duk_and_future['weekofyear'].apply(np.int64)


# In[19]:


# make prediction about the next 24 hours
duk_and_future['prediction'] = reg_new.predict(duk_and_future.drop('DUK_MW',axis = 1))


# In[20]:


# get the demand, prediction_EIA, and prediction of the past 30 days
df = duk_and_future[['DUK_MW','prediction']]
df = df.rename(columns={'DUK_MW':'demand'})
df = df.loc[df.index >= df.index[-1] - timedelta(days=30) ]
df = df.reset_index()
df = df.rename(columns={'index':'Datetime'})
df2 = df2.reset_index()
df = pd.merge(df,df2,on = 'Datetime', how = 'left')
df = df.set_index('Datetime')
df


# In[21]:


# get the latest updated historic weather data of Mecklenburg County
base = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/weatherdata/history?&aggregateHours=1&'
time = 'startDateTime=2023-02-19T00:00:00&endDateTime='+t1+'T24:00:00&'
unit = 'unitGroup=us&contentType=csv&dayStartTime=0:0:00&dayEndTime=0:0:00&'
location = 'location=MecklenburgCounty,NC,US&'
api = 'key=D5FAK4DB3LVUJGWCTFZRXA2T6'
url = base + time + unit + location + api
dfM = pd.read_csv(url)
dfM = dfM[['Date time','Address','Temperature']]
dfM = dfM.set_index('Date time')
dfM


# In[22]:


#get the rest 10 cities historic weather data
city_list = ['GuilfordCounty,NC', 'GreenvilleCounty,SC', 'ForsythCounty,NC', 'SpartanburgCounty,SC', 'DurhamCounty,NC', 'YorkCounty,SC', 'IndianTrail,NC', 'GastonCounty,NC',
              'CabarrusCounty,NC', 'AndersonCounty,SC']
for i in city_list:
    base = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/weatherdata/history?&aggregateHours=1&'
    time = 'startDateTime=2023-02-19T10:00:00&endDateTime='+t1+'T24:00:00&'
    unit = 'unitGroup=us&contentType=csv&dayStartTime=0:0:00&dayEndTime=0:0:00&'
    location = 'location='+i+',US&'
    api = 'key=D5FAK4DB3LVUJGWCTFZRXA2T6'
    url = base + time + unit + location + api
    df0 = pd.read_csv(url)
    df0 = df0[['Date time','Address','Temperature']]
    df0 = df0.set_index('Date time')
    dfM = pd.concat([dfM,df0],axis=1)
    dfM.dropna(inplace = True)


# In[23]:


# get the actual temperature data
TemHistory = dfM['Temperature']
TemHistory['actual_temp'] = dfM.mean(axis=1)
TemHistory = TemHistory[['actual_temp']]
TemHistory.index.name = 'Datetime_old'
TemHistory = TemHistory.reset_index()
TemHistory['Datetime'] = pd.to_datetime(TemHistory['Datetime_old'],format='%m/%d/%Y %H:%M:%S')
TemHistory = TemHistory[['Datetime','actual_temp']]
TemHistory = TemHistory.set_index('Datetime')
TemHistory


# In[24]:


# categorize the rest to be the forecasted temperature
forecasted_temp_new = duk_and_future.loc[duk_and_future.index > TemHistory.index[-1],][['avg_temp']]
forecasted_temp_new = forecasted_temp_new.rename(columns = {'avg_temp':'forecasted_temp'})
forecasted_temp_new


# In[25]:


df.index[-1]-timedelta(days=30)


# In[26]:


df = pd.concat([df,TemHistory],axis=1)


# In[27]:


df = pd.concat([df,forecasted_temp_new],axis=1)


# In[28]:


# load the past 30days of dataframe for later plotly
df = df[df.index >= (df.index[-1]-timedelta(days=30))]
df


# In[29]:


#Calculate 2023 peak
annual_peak_2023 = duk_and_future.loc[duk_and_future.index>='2023-01-01'].DUK_MW.max()
annual_peak_2023


# In[30]:


duk_annual_peak = duk_and_future.loc[duk_and_future.index>='2023-01-01'].dropna().sort_values(by = 'DUK_MW',ascending = False).head(10)
duk_annual_peak = duk_annual_peak[['DUK_MW']]
duk_annual_peak.style.set_caption('DUK 2023 top10 demand')


# In[31]:


# Use Plotly to show final results

import plotly.graph_objects as go
from plotly.subplots import make_subplots



fig = make_subplots(specs=[[{"secondary_y": True}]])



fig.add_trace(go.Scatter(x=df.index, y=df.demand, name='demand',yaxis="y3",
                         line = dict(color='royalblue', width=4)))

fig.add_trace(go.Scatter(x=df.index, y=df.prediction,visible='legendonly',name='prediction',yaxis="y3",
                         line = dict(color='lime', width=4, dash='dot')))

fig.add_trace(go.Scatter(x=df.index, y=df.prediction_EIA,visible='legendonly',name='prediction_EIA',yaxis="y3",
                         line=dict(color='mediumpurple', width=4,dash='dot')))


fig.add_trace(go.Scatter(x=df.index, y=df.actual_temp,name='actual_temp',yaxis="y1",
                         line=dict(color='rosybrown', width=4)))


fig.add_trace(go.Scatter(x=df.index, y=df.forecasted_temp,visible='legendonly',name='forecasted_temp',yaxis='y1',
                         line=dict(color='red', width=4)))


# fig.add_hline(y = 40, line_dash="dot",line_color='black',line_width=2,
#               annotation_text="2023 annual peak", 
#               annotation_position="bottom right")

fig.add_shape(type="line",
    x0=df.index[0], y0=annual_peak_2023, x1=df.index[-1], y1=annual_peak_2023,
    line=dict(color="black",width=3,dash="dot",),yref="y3"
)

# fig.add_trace(go.Scatter(
#     x=,
#     y=,
#     text='2023 annual peak',
#     mode="text",
# ))

# fig.update_layout(yaxis=dict(anchor = 'free',overlaying='y',autoshift=True))

fig.update_layout(
#     xaxis=dict(domain=[0.25, 0.75]),
#     yaxis=dict(
#         title="yaxis title",
#     ),
    yaxis3=dict(
        title="DUK demand (megawatthours)",
        overlaying="y",
        side="left",
        range=[8000,20000],
    ),
    yaxis1=dict(
        title="Temperature (°F)",
#         overlaying="y",
        side="right",
        range=[0,84]
    ),
    title='Past 30 days elecrtricity demand'
     
#     yaxis3=dict(title="yaxis3 title", anchor="free", overlaying="y", autoshift=True),
#     yaxis4=dict(
#         title="yaxis4 title",
#         anchor="free",
#         overlaying="y",
#         autoshift=True,
#         shift=-100,
#     ),
)

fig.update_layout(template=TEMPLATE)

fig.show()

# fig.update_xaxes(title_text="Next 24 hours electricity demand prediction for Duke Energy Carolinas(DUK)")
# fig.update_xaxes(title_text="Date")
# fig.update_yaxes(title_text="DUK demand (megawatthours)", secondary_y=False)
# fig.update_yaxes(title_text="Temperature (°F)", secondary_y=True)


# In[32]:


df_latest_36 = df.loc[df.index >= (df.index[-1] - timedelta(hours=36))]
df_latest_36


# In[33]:


# Use Plotly to show final results

import plotly.graph_objects as go
from plotly.subplots import make_subplots



fig1 = make_subplots(specs=[[{"secondary_y": True}]])



fig1.add_trace(go.Scatter(x=df_latest_36.index, y=df_latest_36.demand, name='demand',yaxis="y3",
                         line = dict(color='royalblue', width=4)))

fig1.add_trace(go.Scatter(x=df_latest_36.index, y=df_latest_36.prediction, name='prediction',yaxis="y3",
                         line = dict(color='lime', width=4, dash='dot')))

fig1.add_trace(go.Scatter(x=df_latest_36.index, y=df_latest_36.prediction_EIA,visible='legendonly',name='prediction_EIA',yaxis="y3",
                         line=dict(color='mediumpurple', width=4,dash='dot')))


fig1.add_trace(go.Scatter(x=df_latest_36.index, y=df_latest_36.actual_temp,visible='legendonly',name='actual_temp',yaxis="y1",
                         line=dict(color='rosybrown', width=4)))


fig1.add_trace(go.Scatter(x=df_latest_36.index, y=df_latest_36.forecasted_temp,name='forecasted_temp',yaxis='y1',
                         line=dict(color='red', width=4)))


fig1.add_hline(y = 70, line_dash="dot",line_color='black',line_width=0,
              annotation_text="2023 annual demand peak", 
              annotation_position="bottom right")

fig1.add_shape(type="line",
    x0=df_latest_36.index[0], y0=annual_peak_2023, x1=df_latest_36.index[-1], y1=annual_peak_2023,
    line=dict(color="black",width=3,dash="dot",),yref="y3"
)

# fig.add_trace(go.Scatter(
#     x=,
#     y=,
#     text='2023 annual peak',
#     mode="text",
# ))

# fig.update_layout(yaxis=dict(anchor = 'free',overlaying='y',autoshift=True))

fig1.update_layout(
#     xaxis=dict(domain=[0.25, 0.75]),
#     yaxis=dict(
#         title="yaxis title",
#     ),
    yaxis3=dict(
        title="DUK demand (megawatthours)",
        overlaying="y",
        side="left",
        range=[8000,20000],
    ),
    yaxis1=dict(
        title="Temperature (°F)",
#         overlaying="y",
        side="right",
        range=[0,84]
    ),
#     title= 'CURRENT RUN: ' + str(date.today()) + ' ' + str(pd.Timestamp.now().hour) + ':00'
    
     
#     yaxis3=dict(title="yaxis3 title", anchor="free", overlaying="y", autoshift=True),
#     yaxis4=dict(
#         title="yaxis4 title",
#         anchor="free",
#         overlaying="y",
#         autoshift=True,
#         shift=-100,
#     ),
)

fig1.update_layout(template=TEMPLATE)

fig1.show()

# fig.update_xaxes(title_text="Next 24 hours electricity demand prediction for Duke Energy Carolinas(DUK)")
# fig.update_xaxes(title_text="Date")
# fig.update_yaxes(title_text="DUK demand (megawatthours)", secondary_y=False)
# fig.update_yaxes(title_text="Temperature (°F)", secondary_y=True)


# In[34]:


# get the future 15 days weather forecast of spartanburg,sc
url = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/spartanburg,sc?unitGroup=us&include=days&key=D5FAK4DB3LVUJGWCTFZRXA2T6&contentType=csv'
dfM1 = pd.read_csv(url)


# In[35]:


# select the tempmax and tempmin from the weather data
dfM1 = dfM1[['datetime','name','tempmax','tempmin']]
dfM1 = dfM1.set_index('datetime')
dfM1


# In[36]:


city_list = ['GuilfordCounty,NC', 'GreenvilleCounty,SC', 'ForsythCounty,NC', 'SpartanburgCounty,SC', 'DurhamCounty,NC', 'YorkCounty,SC', 'IndianTrail,NC', 'GastonCounty,NC',
              'CabarrusCounty,NC', 'AndersonCounty,SC']

for i in city_list:
    url = url = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/'+i+'?unitGroup=us&include=days&key=D5FAK4DB3LVUJGWCTFZRXA2T6&contentType=csv'
    df0 = pd.read_csv(url)
    df0 = df0[['datetime','name','tempmax','tempmin']]
    df0 = df0.set_index('datetime')
    dfM1 = pd.concat([dfM1,df0],axis=1)
    dfM1.dropna(inplace = True)


# In[37]:


# get the 15 day weather forecast for duke energy carolinas
Tem15d = dfM1[['tempmax','tempmin']]
Tem15d['temp_max']= Tem15d.tempmax.mean(axis=1)
Tem15d['temp_min']= Tem15d.tempmin.mean(axis=1)
Tem15d = Tem15d[['temp_max','temp_min']]
Tem15d


# In[38]:


fig2 = make_subplots(specs=[[{"secondary_y": True}]])
fig2.add_trace(go.Scatter(x=Tem15d.index, y=Tem15d.temp_max,name='temp_max',yaxis="y1",
                         line=dict(color='red', width=4)))


fig2.add_trace(go.Scatter(x=Tem15d.index, y=Tem15d.temp_min,name='temp_min',yaxis='y1',
                         line=dict(color='rosybrown', width=4)))

fig2.update_layout(template=TEMPLATE)

fig2.update_layout(title = 'Next 15 days average weather forecast') 

fig2.show()


# In[39]:


duk_annual_peak = duk_annual_peak.reset_index()


# In[40]:


duk_annual_peak.rename(columns={'index':'Datetime','DUK_MW':'Actual_Demand'},inplace=True)


# In[41]:


duk_annual_peak.index = range(1,11)


# In[42]:


print(range(10,0))


# In[43]:


duk_annual_peak


# In[44]:


fig3 = px.bar(duk_annual_peak, x='Actual_Demand', y=duk_annual_peak.index,
              orientation = 'h',text = 'Datetime'
             )
fig3.update_layout(yaxis= dict(title = 'ranking'))

fig3.update_layout(xaxis= dict(title = 'demand'))

fig3.update_layout(template=TEMPLATE)

fig3.update_traces(opacity=0.75)

fig3.update_layout(title = 'Top 10 demands of year 2023 ') 

fig3.show()


# In[45]:


description = 'Duke Energy Carolinas is a subsidiary of Duke Energy, one of the largest electric power holding companies in the United States.Duke Energy Carolinas serves approximately 2.6 million customers in North Carolina and South Carolina.The company provides electric service to residential, commercial, and industrial customers,as well as wholesale customers such as municipalities and electric cooperatives.'

model_description= 'The DUK forecasting model was trained on historical load and weather datafrom 2015/7-2023/2. Weather readings were from VisualCrossing.'


# In[46]:


app = dash.Dash(
    external_stylesheets=[dbc.themes.LUX],
    suppress_callback_exceptions=True
)
server = app.server


# In[47]:


app = dash.Dash(
    external_stylesheets=[dbc.themes.LUX],
    suppress_callback_exceptions=True
)
server = app.server


# In[48]:


app.layout = html.Div([
        html.Div(id='duk-content'),
        html.Br(),
#         dbc.Row([
#             dbc.Col(
#                 html.Div(l.BUTTON_LAYOUT), width=4),
#             dbc.Col(width=7),
#         ], justify='center'),
#     html.Br(),
#         html.Br(),
        dbc.Row([
            dbc.Col(html.H1('Duke Energy Carolinas (DUK)'), width=9),
            dbc.Col(width=2),
        ], justify='center'),
    dbc.Row([
            dbc.Col(
            html.Div(children=description), width=9),
            dbc.Col(width=2)
        ], justify='center'),
    html.Br(),
        dbc.Row([
            dbc.Col(
                html.H3('DUK electricity demand prediction'), width=9
            ),
            dbc.Col(width=2),
        ], justify='center'),
        dbc.Row([
            dbc.Col(
            html.Div('CURRENT RUN: ' + str(date.today()) + ' ' + str(pd.Timestamp.now().hour) + ':00'), width=9),
            dbc.Col(width=2)
        ], justify='center'),
#         dbc.Row([
#             dbc.Col(
#                 html.Div(
#                     children='Mean Absolute Error (MAE)'
#                 ), width=9
#             ),
#             dbc.Col(width=2),
#         ], justify='center'),
#      html.Br(),
#         dbc.Row([
#             dbc.Col(
#                     dcc.Dropdown(
#                         id='duk-dropdown',
#                         options=[
#                             {'label': 'Actual', 'value': 'Actual'},
#                             {'label': 'Predicted', 'value': 'Predicted'}
#                         ],
#                         value=['Actual', 'Predicted'],
#                         multi=True,
#                     ), width=6
#             ),
#             dbc.Col(width=5),
#         ], justify='center'),
    dcc.Graph(id='duk-graph',
             figure=fig1),
        html.Br(),
        dbc.Row([
            dbc.Col(html.H3('Training Data'), width=9),
            dbc.Col(width=2)
        ], justify='center'),
        dbc.Row([
            dbc.Col(
                    html.Div(children=model_description), width=9
            ),
            dbc.Col(width=2)
        ], justify='center'),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    html.Div([
                        dcc.Graph(
                            figure=fig3
                        ),
                    ]), width=4),
                dbc.Col(
                    html.Div([
                        dcc.Graph(
                            figure=fig2
                        ),]), width=4),
                dbc.Col(
                    html.Div([
                        dcc.Graph(
                            figure=fig
                        ),]), width=4)
])                
])
        


# In[49]:


if __name__ == '__main__':
    app.run_server(debug = True,use_reloader=False)

