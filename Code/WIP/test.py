# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 12:17:26 2020

@author: Vikram.V.Kumar
"""
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

import streamlit as st
import pandas as pd
import numpy as np
import time
import pydeck as pdk
from PIL import Image
image = Image.open('C:\\\\Users\\Rupesh.Dahuja\\Desktop\\Hackathon 2020\\Data\\Shell_Image.jpg')
st.image(image)
agree = st.button("About")
if agree: st.write("This is a Streamlit Web report for Hackathon from training data") 
charging_stations = pd.read_csv("C:\\Users\\Rupesh.Dahuja\\Desktop\\Hackathon 2020\\Data\\Shell_Stations.csv")
charging_stations= pd.DataFrame(charging_stations)
charging_stations['lat'] = charging_stations['Latitude']
charging_stations['lon'] = charging_stations['Longitute']

charging_stations1 = pd.read_csv("C:\\Users\\Rupesh.Dahuja\\Desktop\\Hackathon 2020\\Data\\Shell_Stations.csv")
charging_stations1= pd.DataFrame(charging_stations1)
df = pd.DataFrame(charging_stations1,columns = ['Station_ID' ])


#def load_data(nrows):data = pd.read_csv("C:\\Users\\Rupesh.Dahuja\\Desktop\\Hackathon 2020\\Data\\Shell_Stations.csv",nrows=nrows) 
#weekly_data = load_data(1000)
#st.write(weekly_data)
option_city = st.selectbox( "Which City would you like to Analyze?", charging_stations.Place.unique())

option_company  = st.selectbox( "Which Company would you like to Analyze?", charging_stations.Company.unique())
charging_stations = charging_stations.loc[charging_stations['Company'] == option_company]

#st.subheader('Map of all Gas & Electric Vehicles Charging Stations')



genre = st.radio(
   "Which report you would like to see ",
  ('Capacity vs Requirement report', 'Stations recommendation report'))

if genre == 'Capacity vs Requirement report':st.pydeck_chart(pdk.Deck(
     map_style='mapbox://styles/mapbox/light-v9',

     initial_view_state=pdk.ViewState(
         latitude=26.1224,
         longitude=-80.1373,
         zoom=11,
         pitch=50,
     ),
  
 
     layers=[             
            
         pdk.Layer(

             'ScatterplotLayer',
#             'PolygonLayer, ScatterplotLayer' ,'GreatCircleLayer', S2Layer','TextLayer',
             data= charging_stations,
             get_position='[lon, lat]',
             get_color='[200, 30, 0, 200]',
             get_radius=300,
 #            display_data = ['city'],
         ), 
                 
      ],
             

 ))
else :st.bar_chart(df)
st.write(df)





