Код переделан под conda

 coords.csv в папку ./results - общий файл, дописывают строки, колонки layout_name,crop_name,ul,ur,br,bl,crs,start,end
 
Результат обработки записывается в coords.csv файл с колонками: 
«layout_name» имя подложки,
«crop_name» имя снимка,  
«ul», «ur», «br», «bl», где лево-верх, право-верх, право-низ, лево-низ координаты, 
«crs» координатная система в формате «EPSG:{12345}», 
«start» и «end» время в формате «%Y-%m-%dT%h:%m:%s» начала и окончания обработки единичного загруженного снимка, после загрузки подложки и снимка, для точного контроля времени обработки."
