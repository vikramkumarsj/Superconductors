# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 12:17:26 2020

@author: Vikram.V.Kumar
"""

# Import Libraries

import streamlit as st
import pandas as pd
import numpy as np
import time
import pydeck as pdk
from PIL import Image
import matplotlib.pyplot as plt
import pycaret
from pycaret.regression import *

Model = load_model("C:/Users/Vikram.V.Kumar/Desktop/Advanced Analytics Practice/Hackathon/Model/GBR_VK")

image = Image.open('C:\\\\Users\\Vikram.V.Kumar\\Desktop\\Advanced Analytics Practice\\Hackathon\\Data\\Shell_Image.jpg')
st.image(image)
st.title("Digital Insights for grid connected electric vehicle charging stations") 


# Read Charging Stations Data
charging_stations = pd.read_csv("C:\\Users\\Vikram.V.Kumar\\Desktop\\Advanced Analytics Practice\\Hackathon\\Data\\Charging_Stations.csv")
charging_stations= pd.DataFrame(charging_stations)

# Read electric Vehicles data
electric_vehicles = pd.read_csv("C:\\Users\\Vikram.V.Kumar\\Desktop\\Advanced Analytics Practice\\Hackathon\\Data\\Electric_Vehicle_Charging_Data.csv")
electric_vehicles = pd.DataFrame(electric_vehicles)

# Select data for analysis

option_city = st.selectbox( "Which City would you like to Analyze?", charging_stations.Place.unique())
option_company  = st.selectbox( "Which Company would you like to Analyze?", charging_stations.Company.unique())
timestamp = st.sidebar.number_input("Status @ Hours", min_value=1, value = 1)

#Filter data for analysis

charging_stations = charging_stations.loc[charging_stations['Place'] == option_city]
charging_stations_display = charging_stations.loc[charging_stations['Company'] == option_company]
length= int(len(charging_stations_display)/2)


#charging_stations = charging_stations.loc[charging_stations.Company.isin(option_company)]

electric_vehicles = electric_vehicles.loc[electric_vehicles['city'] == option_city]
electric_vehicles = electric_vehicles.loc[electric_vehicles['point_in_time'] == timestamp ]

#Read Battery status data
electric_vehicles_Low_Battery = electric_vehicles.loc[electric_vehicles['battery_level'] <= 33]





# Load the model and Prepare data for model Inference


charging_stations['city'] = charging_stations['Place']
ElecVeh_ChargeData  = electric_vehicles
#ElecVeh_ChargeData.columns
df = pd.merge(charging_stations,ElecVeh_ChargeData,on='city',how='inner')
#df.head(2)
#len(df)
df = df.dropna()
#len(df)

#df.shape






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
   df['dist']=haversine_distance(df['latitude_x'],df['longitude_x'], df['latitude_y'],df['longitude_y'])

#len(df)

df['company_station_city_vehile_dist_min']=0
df['sel']=0
del df['timestamp']

##SELECTING MINIMUM DISTANCE OF VEHICLE FROM STATION - 1 FROM EACH COMPANY AT EACH PIT
df['company_station_city_vehile_dist_min'] = df.groupby(['Company', 'city', 'Country','vehicle_id', 'point_in_time'])['dist'].transform('min')
df.loc[df['dist'].eq(df['company_station_city_vehile_dist_min']),'sel']=1
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

##### UPDATE THE OTHER CHARGING STATIONS NEARBY ############  
df.rename(columns = {'station_competitor':'Other_Charging_Station_Nearby_Flag'},inplace=True)
df_final = df.groupby(['Country','city','Company','Station_ID','point_in_time','latitude_x','longitude_x','Other_Charging_Station_Nearby_Flag','Peak_hour','flag_cold','flag_weekday','Vehicle_count_city','flag_peak','Vehicle_count_@station']).agg({'cum_battery_less_33_@station':'min','cum_battery_more_33_@station':'min','cum_battery_all_@station':'min','cum_vehicles_less_33_@station':'min','cum_vehicles_more_33_@station':'min'}).reset_index()

#df_final.to_csv("C:\\Users\\Vikram.V.Kumar\\Desktop\\df_final.csv")

df_model = df_final.reset_index()
Charge_Status = df_model
Charge_Status = Charge_Status.loc[Charge_Status['city'] == option_city]
Charge_Status = Charge_Status.loc[Charge_Status['Company'] == option_company]
Charge_Status = Charge_Status.loc[Charge_Status['point_in_time'] == timestamp]

Charge_Status['Station Latitude'] = Charge_Status['latitude_x']
Charge_Status['Station Longitute'] = Charge_Status['longitude_x']

Charge_Status = Charge_Status[['Station Latitude', 'Station Longitute', 'point_in_time',
       'Other_Charging_Station_Nearby_Flag', 'Peak_hour', 'flag_cold',
       'flag_weekday', 'Vehicle_count_city', 'flag_peak',
       'Vehicle_count_@station', 'cum_vehicles_less_33_@station',
       'cum_vehicles_more_33_@station']]

Charge_Status = Charge_Status.fillna(1)

# Load model and apply the data

Charge_Status = predict_model(Model, data=Charge_Status)

#Merge for all stations in the city
Charge_Status = pd.merge(charging_stations_display, Charge_Status,  how='left', left_on=['latitude',  'longitude']
                           , right_on = ['Station Latitude',  'Station Longitute'])
Charge_Status['Label'] = Charge_Status['Label'].fillna(0)
Charge_Status['point_in_time'] = Charge_Status['point_in_time'].fillna(timestamp)


#Charge_Status.head(2)
# Options to select Report

genre = st.radio(
   "Which Insights & report you would like to see ",
  ('Display charging stations data along with electric vehicles', 
   'Display of electric vehicles along with those with less Battery Status', 
   'Electric Power Demand prediction','Trend reports'))

if genre == 'Display charging stations data along with electric vehicles':

 
      # Display charging stations data
     st.subheader('Data & Mapping of all Gas & Electric Vehicles Charging Stations')
     st.dataframe(charging_stations_display)    
     st.pydeck_chart(pdk.Deck(
     map_style='mapbox://styles/mapbox/light-v9',
     

     initial_view_state=pdk.ViewState(
#         latitude= 26.1224,
#         longitude=-80.1373,
         latitude= charging_stations_display['latitude'].iloc[length],
         longitude= charging_stations_display['longitude'].iloc[length],
         zoom=11,
         pitch=50,
     ),
  
     layers=[             
            
         pdk.Layer(

             'ScatterplotLayer',
#             'PolygonLayer, ScatterplotLayer' ,'GreatCircleLayer', S2Layer','TextLayer',
             data= charging_stations_display,
             get_position='[longitude, latitude]',
             get_color='[200, 30, 0, 200]',
             get_radius=300,
 #            display_data = ['city'],
         ), 
                 
      ],
             

 ))

#if genre == 'Display electric vehicles' :

  
 
    #Display electric vehicles
     st.subheader('Data and Mapping of electric vehicle charging Stations with Electric vehicle density')
     st.dataframe(electric_vehicles)
    
     st.pydeck_chart(pdk.Deck(
     map_style='mapbox://styles/mapbox/light-v9',
     
     initial_view_state=pdk.ViewState(
         latitude= charging_stations_display['latitude'].iloc[length],
         longitude= charging_stations_display['longitude'].iloc[length],
         zoom=11,
         pitch=50,
     ),
 
     layers=[             

          pdk.Layer(

             'ScreenGridLayer',
#             'PolygonLayer, ScatterplotLayer' ,'GreatCircleLayer', S2Layer','TextLayer',
             data= charging_stations_display,
             get_position='[longitude, latitude]',
             get_color='[200, 30, 0, 200]',
             get_radius=300,
 #            display_data = ['city'],
         ), 
                 
         pdk.Layer(
            'HexagonLayer',
             data=electric_vehicles,
            get_position='[longitude, latitude]',
            radius=100,
            elevation_scale=1,
            elevation_range=[0, 1000],
            pickable=True,
            extruded=True,

         ), 
                 
#         pdk.Layer(
#
#             'ScatterplotLayer',
##             'PolygonLayer, ScatterplotLayer' ,'GreatCircleLayer', S2Layer','TextLayer',
#             data= charging_stations_display,
#             get_position='[longitude, latitude]',
#             get_color='[200, 30, 0, 200]',
#             get_radius=300,
# #            display_data = ['city'],
#         ), 
                 
     ],
             

 ))





if genre == 'Display of electric vehicles along with those with less Battery Status':
    
    # Display low power battery electric vehicles 
    st.subheader('Insights of Actual Demand: Map of electric vehicles with Less Battery Status')
    st.dataframe(electric_vehicles_Low_Battery)

    st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',

     initial_view_state=pdk.ViewState(
         latitude= charging_stations_display['latitude'].iloc[length],
         longitude= charging_stations_display['longitude'].iloc[length],
         zoom=11,
         pitch=50,
     ),
  
 
     layers=[  
             
         pdk.Layer(
            'HexagonLayer',
             data=electric_vehicles_Low_Battery,
            get_position='[longitude, latitude]',
            radius=100,
            elevation_scale=1,
            elevation_range=[0, 1000],
            pickable=True,
            extruded=True,

         ),
            
        pdk.Layer(
               'TextLayer',
                data= charging_stations_display,
                pickable=True,
                get_position='[longitude, latitude]',
                get_text='Station_ID',
                get_size=32,
                get_angle=0,
                get_text_anchor='middle',
        ),

 
     ],
             

 ))
         

     #Display electric vehicles
    st.subheader('Data and Mapping of electric vehicle charging Stations with Electric vehicle density')
    st.dataframe(electric_vehicles)
    
    st.pydeck_chart(pdk.Deck(
     map_style='mapbox://styles/mapbox/light-v9',
     
     initial_view_state=pdk.ViewState(
         latitude= charging_stations_display['latitude'].iloc[length],
         longitude= charging_stations_display['longitude'].iloc[length],
         zoom=11,
         pitch=50,
     ),
 
     layers=[             

          pdk.Layer(

             'ScreenGridLayer',
#             'PolygonLayer, ScatterplotLayer' ,'GreatCircleLayer', S2Layer','TextLayer',
             data= charging_stations_display,
             get_position='[longitude, latitude]',
             get_color='[200, 30, 0, 200]',
             get_radius=300,
 #            display_data = ['city'],
         ), 
                 
         pdk.Layer(
            'HexagonLayer',
             data=electric_vehicles,
            get_position='[longitude, latitude]',
            radius=100,
            elevation_scale=1,
            elevation_range=[0, 1000],
            pickable=True,
            extruded=True,

         ),  
     ],
             

 ))
    

     
if genre == 'Electric Power Demand prediction':
    
    #  Calculate required Power in the area for elecric vehicles
    st.subheader('Eletric Power Demand prediction around the charging station area in KW')
    Charge_Status['lat'] = Charge_Status['latitude']
    Charge_Status['lon'] = Charge_Status['longitude']
    Charge_Status['Label'] = Charge_Status['Label'].replace(np.nan, 0)
    Charge_Status['Label'] = Charge_Status['Label'].round(decimals=2)
    Charge_Status['Power_Requirement'] = Charge_Status['Label'].astype(str) 
    Charge_Status = Charge_Status[['Place','Station_ID','lat','lon','Other_Charging_Station_Nearby_Flag_x','Power_Requirement']]
    
    st.dataframe(Charge_Status)
    length1= int(len(Charge_Status)/2)
        
    st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',

     initial_view_state=pdk.ViewState(
         latitude= Charge_Status['lat'].iloc[length1],
         longitude= Charge_Status['lon'].iloc[length1],
         zoom=11,
         pitch=50,
     ),

     layers=[             

  
         pdk.Layer(

             'ScreenGridLayer',
#             'PolygonLayer, ScatterplotLayer' ,'GreatCircleLayer', S2Layer','TextLayer',
             data= Charge_Status,
             get_position='[lon, lat]',
             get_color='[200, 30, 0, 200]',
             get_radius=50,
 #            display_data = ['city'],
         ), 
            
        pdk.Layer(
               'TextLayer',
                data= Charge_Status,
                pickable=True,
                get_position='[lon, lat]',
                get_text='Power_Requirement',
                get_size=20,
                get_angle=0,
                get_text_anchor='middle',
        ),

      ],

))
     
    # Display low power battery electric vehicles 
    st.subheader('Insights of Actual Demand: Map of electric vehicles with Less Battery Status')
    st.dataframe(electric_vehicles_Low_Battery)

    st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',

     initial_view_state=pdk.ViewState(
         latitude= charging_stations_display['latitude'].iloc[length],
         longitude= charging_stations_display['longitude'].iloc[length],
         zoom=11,
         pitch=50,
     ),
  
 
     layers=[  
             
         pdk.Layer(
            'HexagonLayer',
             data=electric_vehicles_Low_Battery,
            get_position='[longitude, latitude]',
            radius=100,
            elevation_scale=1,
            elevation_range=[0, 1000],
            pickable=True,
            extruded=True,

         ),
            
        pdk.Layer(
               'TextLayer',
                data= charging_stations_display,
                pickable=True,
                get_position='[longitude, latitude]',
                get_text='Station_ID',
                get_size=32,
                get_angle=0,
                get_text_anchor='middle',
        ),

 
     ],
             

 ))
     
if genre == 'Trend reports':   
#  Demand vs Capacity trend report for Grid optimization
     
    
    option_Station = st.selectbox( "Which Station would you like to Analyze?", charging_stations.Station_ID.unique())

    Demand_Trend = pd.read_csv("C:\\Users\\Vikram.V.Kumar\\Desktop\\Advanced Analytics Practice\\Hackathon\\Data\\Demand_Trend.csv")
    Demand_Trend = Demand_Trend.loc[Demand_Trend['Place'] == option_city]
   
    st.subheader('Capacity @ Stations vs Predicted Demand Trend report')
    Demand_Trend1 = pd.DataFrame(Demand_Trend[:], columns = ["Capacity","Predicted Demand"])
    #Line Chart
    st.line_chart(Demand_Trend1) 

    
    st.subheader('Capacity @ Stations vs Actual Demand Trend report')
    Demand_Trend2 = pd.DataFrame(Demand_Trend[:], columns = ["Capacity","Actual Demand"])  
    #Line Chart
    st.line_chart(Demand_Trend2) 

    
    st.subheader('Predicted Demand vs Actual Demand Trend report')   
    Demand_Trend3 = pd.DataFrame(Demand_Trend[:], columns = ["Predicted Demand","Actual Demand"])
    #Line Chart
    st.line_chart(Demand_Trend3)    

    Demand_Trend4 = pd.DataFrame(Demand_Trend[:10], columns = ["Actual Demand","Predicted Demand"])    
    #histogram
    Demand_Trend4.hist()
    plt.show()
    st.pyplot()   
    
    #Area Chart
    Demand_Trend5 = pd.DataFrame(Demand_Trend[:], columns = ["Actual Demand","Predicted Demand"])  
    st.area_chart(Demand_Trend5)


else:  print('exit')





