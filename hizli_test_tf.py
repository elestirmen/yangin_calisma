from osgeo import gdal
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model    

gdal.AllRegister()

# Modeli önceden yükle
model = load_model("eniyi_model.h5")

# Test dosya dizini
test_dir = r"test"

for file in os.listdir(test_dir):
    filepath = os.path.join(test_dir, file)
    print(filepath)
    
    # Giriş raster verisini açma
    ds = gdal.Open(filepath)
    count = ds.RasterCount
    
    # İlk bandın boyutlarını belirleme
    band = ds.GetRasterBand(1)
    rows, cols = band.YSize, band.XSize
    
    # Tüm bandları bir kerede okuma (vstack yerine daha hızlı alternatif)
    yeni_arr = np.array([ds.GetRasterBand(b).ReadAsArray() for b in range(1, 13)])
    yeni_arr = np.moveaxis(yeni_arr, 0, -1)  # Band eksenini son eksene taşı
    
    # Normalizasyon
    normal_yeni_arr = yeni_arr.reshape(-1, 12)

    # Model tahminlerini toplu olarak yapma
    tahminler = model.predict(normal_yeni_arr, batch_size=1024)  # Daha büyük batch size ile hız artar
    ikiboyuttahmin = tahminler.reshape(rows, cols)
    
    # 4095 değerini hızlıca atama (vektörleştirilmiş numpy işlemi)
    ikiboyuttahmin = np.where(ikiboyuttahmin > 0.5, 4095, 0)
    
    # Sonuçları görselleştirme
    plt.figure()
    plt.imshow(ikiboyuttahmin)
    plt.show()
    
    # Raster bilgilerini çıktı olarak gösterme
    print(f"Raster shape: {rows}x{cols}, Band count: {count}")
    print(f"Size: {band.XSize} x {band.YSize}")
    print(f"Max: {band.GetMaximum()}, Min: {band.GetMinimum()}")
    print(f"Data Type: {gdal.GetDataTypeName(band.DataType)}")
