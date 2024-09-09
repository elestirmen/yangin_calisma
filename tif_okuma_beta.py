from osgeo import ogr, gdal, osr
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings
import seaborn as sbn

warnings.filterwarnings('ignore')


gdal.AllRegister()


df11=pd.DataFrame()
df22=pd.DataFrame()

veri=""

file = r"yanmamis"

satir = []
for veri in os.listdir(file):
    file = r"yanmamis"
    file = os.path.join(file,veri) 
    
    
    print(file)
    
    (fileRoot, fileExt) = os.path.splitext(file)
    outFileName = fileRoot + "_mod" + fileExt
    
    
    sutun = []  
    
    
    ds = gdal.Open(file)
    count = ds.RasterCount
    

    
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    [rows, cols] = arr.shape
    

    
    print(arr.shape)
    print(count)
    
    
    
    yeni_arr=np.empty((rows,cols,13))
    
    b=1
    while b<13:
        i=0
        band = ds.GetRasterBand(b)
        arr = band.ReadAsArray()
        while i<rows:
            j=0
            while j<cols:  
                   
                
                yeni_arr[i,j,b-1] =arr[i][j]  
                  
                j+=1        
                  
            i+=1
            
        b+=1
    yeni_arr[:,:,12]=0
    


    
    yeni_arr_reshaped=yeni_arr.reshape(-1,13)
    
    print(yeni_arr.shape)
    #input("pause")
    
    df1=pd.DataFrame(yeni_arr_reshaped)
    
    df11 = pd.concat([df11, df1])

    #df11=df11.append(df1)  
   
    
#%% YANMIŞ

veri=""
file = r"yanmis"

satir = []
for veri in os.listdir(file):
    file = r"yanmis"
    file = os.path.join(file,veri) 
    
    
    print(file)
    
    (fileRoot, fileExt) = os.path.splitext(file)
    outFileName = fileRoot + "_mod" + fileExt
    
    
    sutun = []  
    
    
    ds = gdal.Open(file)
    count = ds.RasterCount
    

    
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    [rows, cols] = arr.shape
    

    
    print(arr.shape)
    print(count)
    
    
    
    yeni_arr=np.empty((rows,cols,13))
    
    b=1
    while b<13:
        i=0
        band = ds.GetRasterBand(b)
        arr = band.ReadAsArray()
        while i<rows:
            j=0
            while j<cols:  
                   
                
                yeni_arr[i,j,b-1] =arr[i][j]  
                  
                j+=1        
                  
            i+=1
            
        b+=1
    yeni_arr[:,:,12]=1
    
    
    

    
    
   
    
    yeni_arr_reshaped=yeni_arr.reshape(-1,13)
    
    print(yeni_arr.shape)
    #input("pause")
    
    df2=pd.DataFrame(yeni_arr_reshaped)
    
    df22 = pd.concat([df22, df2])
    
    #df22=df22.append(df2)    
   
    
    
frames = [df11, df22]

result = pd.concat(frames)
   
    
    
    
print(df2.shape)
    


import pickle
pickle_out = open("butun.pickle","wb")
pickle.dump(result, pickle_out)
pickle_out.close()
    
pickle_in = open("butun.pickle","rb")
df3 = pickle.load(pickle_in)