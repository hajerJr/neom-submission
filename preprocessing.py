from datetime import datetime
from pysolar.solar import *
import pvlib
import pytz
from pytz import timezone
import numpy as np
import pandas as pd
import math

# Convert Radian Angle to Degree
def degree(x):
    pi=math.pi
    degree=(x*180)/pi
    return degree

df = pd.read_csv("Data.csv")
print(df.head())

# Latitude and Longitude of the preffered location
Latitude = df['Latitude'].iloc[1]
Longitude = df['Longitude'].iloc[1]
print(Latitude)

# Get the dates to iterate through them one by one
dateCol = pd.to_datetime(df['Date'])
print(dateCol)

Dates = dateCol
# Preprocessing missing data in Azimuth column and fill the Zenith Column
for i, a in zip(Dates, range(0, len(Dates))):
    # Get the timedate information seperately for the current row
    Day = i.day
    Month = i.month
    Year = i.year
    Hour = i.hour
    Minutes = i.minute
    Seconds = i.second
    # Set the tzinfo the the current datetime
    fullDate = datetime.datetime(Year, Month, Day, Hour, Minutes, Seconds, tzinfo = pytz.timezone('Asia/Riyadh'))
    # If the current Azimuth value is missing then calculate it using get_azimuth
    if pd.isnull(df['Azimuth Angle (°)'][a]):
        Azimuth = get_azimuth(Latitude, Longitude, fullDate)
        df['Azimuth Angle (°)'][a] = Azimuth
    # Calculate Declination, Hour_angle to get Zenith Angle in Radian
    # Calculate Equation_of_time and Times to get the value of Hour_angle
    Declination = pvlib.solarposition.declination_spencer71(Day)
    Equation_of_time = pvlib.solarposition.equation_of_time_pvcdrom(Day)
    Times = pd.to_datetime([f'{Day}/{Month}/{Year}', np.datetime64(i), datetime.datetime(Year, Month, Day)])
    Hour_angle = pvlib.solarposition.hour_angle(Times, Longitude, Equation_of_time)
    Zenith = pvlib.solarposition.solar_zenith_analytical(Latitude, Hour_angle, Declination)
    # Zenith will be a numpy array, in order to get an int, the average of the three values were taken
    Final_Zenith_value = np.amax(Zenith)
    # Convert from Radian to Degree
    Final_Zenith_value = degree(Final_Zenith_value)
    # Set the value of the current row
    df['Zenith Angle (°)'][a] = Final_Zenith_value
    df['Altitude'][a] = get_altitude(Latitude, Longitude, fullDate)
# Save the Data frame into a .csv file named "Data"
df.to_csv('Data_V2.csv', index=False)
