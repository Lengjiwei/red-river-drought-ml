# download_data.py — placeholder for fetching public datasets
# Edit to implement actual downloads (ERA5-Land/MSWEP links, etc.).
import argparse
from pathlib import Path
import cdsapi
import calendar

c = cdsapi.Client()

dic = {
        'variable':['total_precipitation'],
        'year':'1960',
        'month':'',
        'day':[],
        'time':'00:00',
        'format':'netcdf.zip',
        'area':[,,,],
    }
mon = []
day = []
for m in range(1,13):
    mon.append(str(m))
for d in range(1,32):
    day.append(str(d))

for i in range(1960, 2020): 
    # for j in range(1, 13):  
    # day_num = calendar.monthrange(i, j)[1]  
    dic['year'] = str(i)
    dic['month'] = mon
    dic['day'] = day
    filename = '/home/lin/era5/era5_ppt_' + str(i) + '.zip'  
    c.retrieve('reanalysis-era5-land', dic, filename)  
    print(dic)
print("finish")

