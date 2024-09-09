from osgeo import ogr, gdal, osr
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model    

gdal.AllRegister()

# Model yükleme
model = load_model("eniyi_model.h5")

# Test dosyası
test_dir = r"test"

for file in os.listdir(test_dir):
    filepath = os.path.join(test_dir, file)
    print(filepath)
    
    # Giriş raster verisini açma
    ds = gdal.Open(filepath)
    count = ds.RasterCount
    
    # İlk bandı al ve şekli göster
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    rows, cols = arr.shape
    print(f"Band 1 shape: {arr.shape}, Total bands: {count}")
    
    # Tüm bandları numpy array'e aktarma
    yeni_arr = np.dstack([ds.GetRasterBand(b).ReadAsArray() for b in range(1, 13)])
    print(f"New array shape: {yeni_arr.shape}")
    
    # Normalizasyon
    normal_yeni_arr = yeni_arr.reshape(-1, 12)

    # Model tahminleri
    tahminler = model.predict(normal_yeni_arr)
    ikiboyuttahmin = tahminler.reshape(rows, cols)
    
    # Yanık ve yanmamış alanları sayma
    yanmis = np.count_nonzero(ikiboyuttahmin > 0.5)
    yanmamis = np.count_nonzero(ikiboyuttahmin <= 0.5)
    
    # 4095 değerini atama (döngü yerine numpy ile)
    ikiboyuttahmin = np.where(ikiboyuttahmin > 0.5, 4095, 0)
    
    # Sonuçları görselleştirme
    plt.figure()
    plt.imshow(ikiboyuttahmin)
    plt.show()
    
    # Bilgi çıktısı
    print(f"Raster shape: {arr.shape}, Count: {count}")
    print(f"Size: {band.XSize} x {band.YSize}")
    print(f"Max: {band.GetMaximum()}, Min: {band.GetMinimum()}")
    print(f"Data Type: {gdal.GetDataTypeName(band.DataType)}")
