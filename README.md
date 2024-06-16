# LCT-Ultra-Sitronics
Зависимости
```
git clone --recursive https://github.com/gmberton/image-matching-models
cd image-matching-models
python -m pip install -e .
```
```
rasterio gdal opencv fastapi
```
По условию задачи реализовано два эндпойнта image-match и pixels, которые возвращают текстовые ответы по формату из ТЗ
```
curl -X POST "endless-presently-basilisk.ngrok-free.app/pixels/" -F "file=crop.tif"
```
