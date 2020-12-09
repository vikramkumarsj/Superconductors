# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 13:50:35 2020

@author: Neha.Sharma3
"""

import pandas as pd
import numpy as np


charging_stations = pd.read_csv("C:\\Users\\Neha.Sharma3\\OneDrive\\Documents\\Hackathon\\16Oct\\Data\\Charging_Stations.csv")
ElecVeh_ChargeData = pd.read_csv("C:\\Users\\Neha.Sharma3\\OneDrive\\Documents\\Hackathon\\16Oct\\Data\\Electric_Vehicle_Charging_Data.csv")

charging_stations.columns

df = pd.merge(charging_stations,ElecVeh_ChargeData,on='city',how='inner')
df = df.dropna()
df.to_csv("C:\\Users\\Neha.Sharma3\\OneDrive\\Documents\\Hackathon\\16Oct\\Data\\df.csv")

def haversine_distance(lat1, lon1, lat2, lon2):
   r = 6371
   phi1 = np.radians(lat1)
   phi2 = np.radians(lat2)
   delta_phi = np.radians(lat2 - lat1)
   delta_lambda = np.radians(lon2 - lon1)
   a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
   res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
   return np.round(res, 2)


df['dist'] ='nan'

for row in df.itertuples(index=False):
   df['dist']=haversine_distance(df['Station Latitude'],df['Station Longitute'], df['Vehicle latitude'],df['Vehicle longitude'])

df.to_csv("C:\\Users\\Neha.Sharma3\\OneDrive\\Documents\\Hackathon\\16Oct\\Data\\df_dist.csv")

df.columns

###############################################################################################################################################

df = pd.read_csv("C:\\Users\\Neha.Sharma3\\OneDrive\\Documents\\Hackathon\\16Oct\\Data\\df_dist.csv")

df.columns

df['company_station_city_vehile_dist_min']=0

df['sel']=0

del df['timestamp']

##SELECTING MINIMUM DISTANCE OF VEHICLE FROM STATION - 1 FROM EACH COMPANY AT EACH PIT
df['company_station_city_vehile_dist_min'] = df.groupby(['Company', 'city', 'Country','vehicle_id', 'point_in_time'])['dist'].transform('min')

df.loc[df['dist'].eq(df['company_station_city_vehile_dist_min']),'sel']=1

df.to_csv("C:\\Users\\Neha.Sharma3\\OneDrive\\Documents\\Hackathon\\16Oct\\Data\\df_test.csv")

df= df[df.sel==1]


df['flag_cold'] = np.where(df['temp']<=15,2,1)

df['flag_weekday']= np.where((df['weekday']==6) | (df['weekday']==0), 1, 2)


df['Vehicle_count_city'] = df.groupby(['Country','city','point_in_time'])['vehicle_id'].transform('nunique')


df['flag_peak'] = np.where(df['Vehicle_count_city']>240,2,1)

#battery remaining is the same as battery level

df['battery_level_KW'] = 0
df['battery_level_KW'] = (df['battery_level']/100) *180

df['battery_required_KW'] = df['battery_capacity'] - df['battery_level_KW']

df['flag_battery_less_33_@station'] =0

df['flag_battery_less_33_@station'][df['battery_level']<=33]=1

df['flag_battery_more_33_@station']=0

df['flag_battery_more_33_@station'][df['battery_level']>33]=1

df['Vehicle_count_@station'] = df.groupby(['Country','city','Company','Station_ID','point_in_time'])['vehicle_id'].transform('nunique')


df['cum_battery_less_33_@station'] =0
df['cum_battery_more_33_@station'] =0
df['cum_battery_all_@station'] =0


df['cum_vehicles_less_33_@station'] =0
df['cum_vehicles_more_33_@station'] =0



df['cum_vehicles_less_33_@station'] = df[df['flag_battery_less_33_@station']==1].groupby(['Country','city','Company','Station_ID','point_in_time'])['flag_battery_less_33_@station'].transform('sum')


df['cum_vehicles_more_33_@station'] = df[df['flag_battery_more_33_@station']==1].groupby(['Country','city','Company','Station_ID','point_in_time'])['flag_battery_more_33_@station'].transform('sum')




df['cum_battery_all_@station']  = df.groupby(['Country','city','Company','Station_ID','point_in_time'])['battery_level_KW'].transform('sum')


df['cum_battery_less_33_@station'] = df[df['flag_battery_less_33_@station']==1].groupby(['Country','city','Company','Station_ID','point_in_time'])['battery_level_KW'].transform('sum')


df['cum_battery_more_33_@station'] = df[df['flag_battery_more_33_@station']==1].groupby(['Country','city','Company','Station_ID','point_in_time'])['battery_level_KW'].transform('sum')

df.to_csv("C:\\Users\\Neha.Sharma3\\OneDrive\\Documents\\Hackathon\\16Oct\\Data\\df_agg.csv")


##### UPDATE THE OTHER CHARGING STATIONS NEARBY ############  
df.rename(columns = {'station_competitor':'Other_Charging_Station_Nearby_Flag'},inplace=True)
df.columns


df_final = df.groupby(['Country','city','Station Latitude', 'Station Longitute','Company','Station_ID','point_in_time','Other_Charging_Station_Nearby_Flag','Peak_hour','flag_cold','flag_weekday','Vehicle_count_city','flag_peak','Vehicle_count_@station']).agg({'cum_battery_less_33_@station':'min','cum_battery_more_33_@station':'min','cum_battery_all_@station':'min','cum_vehicles_less_33_@station':'min','cum_vehicles_more_33_@station':'min'})

df_final.to_csv("C:\\Users\\Neha.Sharma3\\OneDrive\\Documents\\Hackathon\\16Oct\\Data\\df_final.csv")

df_final.shape

df_model = df_final.reset_index()

df_model.columns

df_model.shape

#df_model['power_demand'] =0

df_model = df_model.fillna(1)

#df_model['power_demand'] = df_model['cum_battery_less_33_@station'] * (df_model['cum_battery_less_33_@station']/df_model['cum_battery_more_33_@station']) * (df_model['flag_weekday']/2) * (df_model['flag_cold']/2) * (df_model['Other_Charging_Station_Nearby_Flag']/2) * (df_model['Peak_hour']/2)

df_model['power_demand'] = df_model['cum_battery_less_33_@station'] * (df_model['cum_battery_less_33_@station']/df_model['cum_battery_more_33_@station']) 

df_model.to_csv("C:\\Users\\Neha.Sharma3\\OneDrive\\Documents\\Hackathon\\16Oct\\Data\\df_model.csv")

df_model_merge = pd.merge(charging_stations,df_model,on='city',how='inner')


df_model_merge.to_csv("C:\\Users\\Neha.Sharma3\\OneDrive\\Documents\\Hackathon\\16Oct\\Data\\df_model_merge.csv")



##########################################################################################################################################################################################################
#other competitor charging station
##########################################################################################################################################################################################################

charging_stations.columns

ch = pd.read_csv("C:\\Users\\Neha.Sharma3\\OneDrive\\Documents\\Hackathon\\16Oct\\Data\\Charging_Stations.csv")


def haversine_distance(lat1, lon1, lat2, lon2):
   r = 6371
   phi1 = np.radians(lat1)
   phi2 = np.radians(lat2)
   delta_phi = np.radians(lat2 - lat1)
   delta_lambda = np.radians(lon2 - lon1)
   a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
   res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
   return np.round(res, 2)

ch['station_competitor']=2

for i in range(0,106):
    x=ch['Station Latitude']
    x=x.to_frame()
    x['Station Latitude'] = ch['Station Latitude'][i]
    y=ch['Station Longitute']
    y=y.to_frame()
    y['Station Longitute'] = ch['Station Longitute'][i]
    z=ch['Station_ID'][i]
    ch[z]=0
    for row in ch.itertuples(index=False):
        ch[z]=haversine_distance(x['Station Latitude'],y['Station Longitute'], ch['Station Latitude'],ch['Station Longitute'])    
    if(('Shell'not in z) & (ch[z].nsmallest(2).iloc[-1]<1.7)):
        ch['station_competitor'][i]=1
        
ch.to_csv("C:\\Users\\Neha.Sharma3\\OneDrive\\Documents\\Hackathon\\16Oct\\Data\\Charging_Stations_distances.csv")