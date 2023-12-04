import rasterio
from qgis.core import Qgis, QgsRasterBlock


def create_raster_source(pred, profile) -> QgsRasterBlock:
    profile.update(dtype=rasterio.float32, count=1)
    height, width = pred.shape
    profile['width'] = width
    profile['height'] = height
    data_type = Qgis.DataType.Float32
    block = QgsRasterBlock(data_type, width, height)
    block.setData(pred)
    return block
