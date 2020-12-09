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

Model = load_model("Superconductors//Model//GBR_VK")

image = Image.open('/app/Superconductors/Data/Shell_Image.jpg')
st.image(image)
st.title("Digital Insights for grid connected electric vehicle charging stations") 
