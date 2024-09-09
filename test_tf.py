from osgeo import ogr, gdal, osr
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model    

gdal.AllRegister()




veri=""

file = r"test"

satir = []
for veri in os.listdir(file):
    file = r"test"
    file = os.path.join(file,veri) 
    
    
    print(file)
    
    (fileRoot, fileExt) = os.path.splitext(file)
    outFileName = fileRoot + "_mod" + fileExt
    
    
   
    
    
    ds = gdal.Open(file)
    count = ds.RasterCount
    

    
    band = ds.GetRasterBand(1)
    
    arr = band.ReadAsArray()
    print(arr.shape)
    
    
    arr = band.ReadAsArray()
    [rows, cols] = arr.shape
    
   
    
    print(arr.shape)
    print(count)
    
    
    yeni_arr=np.empty((rows,cols,12))
    
    print(yeni_arr.shape)
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
        print(" ",b)   
    
    #%%
    

   

    model=load_model("eniyi_model.h5")

    
    
    
    

  
       
    yanmis=0
    yanmamis=0
    resim=np.empty((rows,cols,1))
    
    #normalization

    normal_yeni_arr = yeni_arr.reshape(-1,12)
    
    
    tahminler=model.predict(normal_yeni_arr)
    
    
    
    #%%
    

   
    ikiboyuttahmin = tahminler.reshape(rows,cols)
    
    yanmis = np.count_nonzero(ikiboyuttahmin >0.5)
    yanmamis = np.count_nonzero(ikiboyuttahmin < 0.5)
    
   
    i=0
    j=0    
    while i<rows:
        j=0
        while j<cols:
            if ikiboyuttahmin[i,j]>0.5:
                ikiboyuttahmin[i,j]=4095
            else:
                 ikiboyuttahmin[i,j]=0
            j+=1
        i+=1
            
   
    plt.figure()
    plt.imshow(ikiboyuttahmin)
    plt.show()
    
    
    
    print(arr.shape)
    print(count)
    print(band.XSize,"X",band.YSize)
    print(band.GetMaximum())
    print(band.GetMinimum())
    print(band.DataType)
 
    
 