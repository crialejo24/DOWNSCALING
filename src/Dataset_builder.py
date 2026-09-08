"""
dataset_builder.py
Construcción de pares temporales Sentinel-2 / Landsat y exportación
del dataset HR/LR (192/48/64) a disco (Drive u otra ruta local).
"""

import os
import random

import cv2
import ee
import geemap
import numpy as np
import rasterio
from rasterio.transform import from_bounds

from .gee_utils import (
    cloud_percentage_s2,
    water_percentage_s2,
    cloud_percentage_landsat,
    variability_s2,
)
from .image_utils import to_uint8_vis, to_uint8_rgb


def nombre_random(n=10):
    """Genera un nombre alfanumérico aleatorio de longitud n."""
    caracteres = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return ''.join(random.choice(caracteres) for _ in range(n))


def get_temporal_pair_rgb(
        start,
        end,
        latitude,
        longitude,
        dim=512,
        max_cloud_pct=5,
        max_water_pct=50,
        day_tolerance=5):
    """
    Busca el mejor par temporal Sentinel-2 / Landsat sobre un punto dado,
    filtrando por nubes, agua y variabilidad espacial, y devuelve las
    imágenes RGB (reflectancia real) listas para exportar.
    """
    px = 10
    dimensions = 1
    d = 0

    while dimensions < dim:
        roi = ee.Geometry.Point([longitude, latitude]).buffer((dim + d) * px / 2).bounds()
        coords = roi.getInfo()['coordinates'][0]
        lon1, lat1 = coords[3]
        lon2, lat2 = coords[1]
        p11 = ee.Geometry.Point(lon1, lat1)
        p12 = ee.Geometry.Point(lon1, lat2)
        p21 = ee.Geometry.Point(lon2, lat1)
        p22 = ee.Geometry.Point(lon2, lat2)

        s2_col = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(p11).filterBounds(p12)
            .filterBounds(p21).filterBounds(p22)
            .filterDate(start, end)
            .map(lambda img: cloud_percentage_s2(img, roi))
            .map(lambda img: water_percentage_s2(img, roi))
            .map(lambda img: variability_s2(img, roi))
            .filter(ee.Filter.lte('CLOUD_PCT_ROI', max_cloud_pct))
            .filter(ee.Filter.lte('WATER_PCT_ROI', max_water_pct))
            .filter(ee.Filter.gte('STD_RGB_ROI', 20))
        )

        roi = ee.Geometry.Rectangle(lon1, lat1, lon2, lat2)

        if s2_col.size().getInfo() == 0:
            print("No hay Sentinel válida")
            return None

        first_img = (
            s2_col.first()
            .select('B2')
            .clip(roi)
            .reproject(crs='EPSG:4326', scale=10)
        )
        info = first_img.getInfo()
        dimensions = min(info['bands'][0]['dimensions'])
        s2_dim = info['bands'][0]['dimensions']
        d += 1

    # ================= LANDSAT =================
    landsat_col = (
        ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
        .filterBounds(p11).filterBounds(p12)
        .filterBounds(p21).filterBounds(p22)
        .filterDate(start, end)
        .map(lambda img: cloud_percentage_landsat(img, roi))
        .filter(ee.Filter.lte('CLOUD_PCT_ROI', max_cloud_pct))
    )

    if landsat_col.size().getInfo() == 0:
        print("No hay Landsat válida")
        return None

    # ================= MATCH TEMPORAL =================
    s2_list = s2_col.toList(s2_col.size())
    landsat_list = landsat_col.toList(landsat_col.size())

    best_pair = None
    min_diff = 9999
    best_s2_date = None
    best_ls_date = None

    for i in range(s2_col.size().getInfo()):
        s2_img_tmp = ee.Image(s2_list.get(i))
        s2_date_tmp = ee.Date(s2_img_tmp.get('system:time_start'))
        for j in range(landsat_col.size().getInfo()):
            ls_img_tmp = ee.Image(landsat_list.get(j))
            ls_date_tmp = ee.Date(ls_img_tmp.get('system:time_start'))
            diff = abs(s2_date_tmp.difference(ls_date_tmp, 'day').getInfo())
            if diff < min_diff:
                min_diff = diff
                best_pair = (s2_img_tmp, ls_img_tmp)
                best_s2_date = s2_date_tmp
                best_ls_date = ls_date_tmp

    if best_pair is None or min_diff > day_tolerance:
        print("No existe par dentro de la tolerancia temporal")
        return None

    print(f"Par encontrado con diferencia de {min_diff} días")

    # ================= IMÁGENES =================
    s2_img, ls_img = best_pair
    s2_image = (
        s2_img.select([
            'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7',
            'B8', 'B8A', 'B11', 'B12'
        ])
        .clip(roi)
        .reproject(crs='EPSG:4326', scale=10)
    )
    landsat_image = (
        ls_img.select([
            'SR_B1', 'SR_B2', 'SR_B3', 'SR_B4',
            'SR_B5', 'SR_B6', 'SR_B7'
        ])
        .clip(roi)
        .reproject(crs='EPSG:4326', scale=30)
    )

    # ===== dimensiones Landsat =====
    first_ls = landsat_image.select('SR_B2')
    ls_info = first_ls.getInfo()
    ls_dim = ls_info['bands'][0]['dimensions']

    # ================= RGB EN REFLECTANCIA REAL =================
    rgb_s2 = s2_image.select(['B4', 'B3', 'B2']).multiply(1e-4)
    rgb_landsat = (
        landsat_image
        .select(['SR_B4', 'SR_B3', 'SR_B2'])
        .multiply(2.75e-5)
        .add(-0.2)
    )

    s2_date_str = best_s2_date.format('YYYY-MM-dd').getInfo()
    ls_date_str = best_ls_date.format('YYYY-MM-dd').getInfo()

    return {
        "rgb_s2": rgb_s2,
        "rgb_landsat": rgb_landsat,
        "s2_full": s2_image,
        "landsat_full": landsat_image,
        "roi": roi,
        "s2_date": s2_date_str,
        "landsat_date": ls_date_str,
        "s2_dimension": s2_dim,
        "landsat_dimension": ls_dim
    }


def save_images(images_dict, nombre, folder, dim=192):
    """
    Guarda las versiones HR (Sentinel) y LR/4, LR/3 (Landsat) del par,
    en formato float, uint8 "modelo" y uint8 "visualizable", además de
    exportar las imágenes completas (todas las bandas) sin recortar a dim.
    """
    # ============================
    # Subcarpetas
    # ============================
    subfolder_1 = "HR_192_norm/"     # HR Sentinel float
    subfolder_2 = "LR_48_normx4/"    # LR/4 Landsat float
    subfolder_3 = "FB_HR_192/"       # Full HR
    subfolder_4 = "FB_LR_48/"        # Full LR
    subfolder_6 = "HR_192_vis/"      # HR visualizable
    subfolder_7 = "LR_48_visx4/"     # LR/4 visualizable
    subfolder_8 = "HR_192_mod/"      # HR para modelo
    subfolder_9 = "LR_48_modx4/"     # LR/4 para modelo
    # Carpetas LR/3
    subfolder_10 = "LR_64_normx3/"   # LR/3 Landsat float
    subfolder_11 = "LR_64_modx3/"    # LR/3 para modelo
    subfolder_12 = "LR_64_visx3/"    # LR/3 visualizable

    for f in [subfolder_1, subfolder_2, subfolder_3, subfolder_4,
              subfolder_6, subfolder_7, subfolder_8, subfolder_9,
              subfolder_10, subfolder_11, subfolder_12]:
        os.makedirs(folder + f, exist_ok=True)

    roi = images_dict["roi"]
    bounds = roi.bounds().getInfo()['coordinates'][0]
    xmin, ymin = bounds[0]
    xmax, ymax = bounds[2]

    # ================= SENTINEL HR =================
    rgb_s2_np = geemap.ee_to_numpy(images_dict["rgb_s2"], region=roi, scale=10).astype(np.float32)
    rgb_s2_resized = cv2.resize(rgb_s2_np, (dim, dim), interpolation=cv2.INTER_AREA).astype(np.float32)
    rgb_s2_resized = np.clip(rgb_s2_resized, 0, 1).astype(np.float32)
    transform_s2 = from_bounds(xmin, ymin, xmax, ymax, dim, dim)

    # HR float
    with rasterio.open(os.path.join(folder + subfolder_1, nombre + "_RGB_S2_float.tif"),
                        'w', driver='GTiff', height=dim, width=dim, count=3,
                        dtype='float32', crs='EPSG:4326', transform=transform_s2) as dst:
        for i in range(3):
            dst.write(rgb_s2_resized[:, :, i], i + 1)

    # HR mod
    rgb_s2_mod = to_uint8_rgb(rgb_s2_resized)
    with rasterio.open(os.path.join(folder + subfolder_8, nombre + "_RGB_S2_mod.tif"),
                        'w', driver='GTiff', height=dim, width=dim, count=3,
                        dtype='uint8', crs='EPSG:4326', transform=transform_s2) as dst:
        for i in range(3):
            dst.write(rgb_s2_mod[:, :, i], i + 1)

    # HR visualizable
    rgb_s2_vis = to_uint8_vis(rgb_s2_resized)
    with rasterio.open(os.path.join(folder + subfolder_6, nombre + "_RGB_S2_vis.tif"),
                        'w', driver='GTiff', height=dim, width=dim, count=3,
                        dtype='uint8', crs='EPSG:4326', transform=transform_s2) as dst:
        for i in range(3):
            dst.write(rgb_s2_vis[:, :, i], i + 1)

    print("Guardado HR Sentinel")

    # ================= LANDSAT LR =================
    rgb_ls_np = geemap.ee_to_numpy(images_dict["rgb_landsat"], region=roi, scale=30).astype(np.float32)
    rgb_ls_np = np.clip(rgb_ls_np, 0, 1).astype(np.float32)

    # --- LR/4 ---
    lr4_dim = int(dim / 4)
    rgb_ls_lr4 = cv2.resize(rgb_ls_np, (lr4_dim, lr4_dim), interpolation=cv2.INTER_AREA)
    transform_ls_lr4 = from_bounds(xmin, ymin, xmax, ymax, lr4_dim, lr4_dim)

    with rasterio.open(os.path.join(folder + subfolder_2, nombre + "_RGB_LS_float_x4.tif"),
                        'w', driver='GTiff', height=lr4_dim, width=lr4_dim, count=3,
                        dtype='float32', crs='EPSG:4326', transform=transform_ls_lr4) as dst:
        for i in range(3):
            dst.write(rgb_ls_lr4[:, :, i], i + 1)

    rgb_ls_lr4_mod = to_uint8_rgb(rgb_ls_lr4)
    with rasterio.open(os.path.join(folder + subfolder_9, nombre + "_RGB_LS_mod_x4.tif"),
                        'w', driver='GTiff', height=lr4_dim, width=lr4_dim, count=3,
                        dtype='uint8', crs='EPSG:4326', transform=transform_ls_lr4) as dst:
        for i in range(3):
            dst.write(rgb_ls_lr4_mod[:, :, i], i + 1)

    rgb_ls_lr4_vis = to_uint8_vis(rgb_ls_lr4)
    with rasterio.open(os.path.join(folder + subfolder_7, nombre + "_RGB_LS_vis_x4.tif"),
                        'w', driver='GTiff', height=lr4_dim, width=lr4_dim, count=3,
                        dtype='uint8', crs='EPSG:4326', transform=transform_ls_lr4) as dst:
        for i in range(3):
            dst.write(rgb_ls_lr4_vis[:, :, i], i + 1)

    print("Guardado LR/4 Landsat")

    # --- LR/3 ---
    lr3_dim = int(dim / 3)
    rgb_ls_lr3 = cv2.resize(rgb_ls_np, (lr3_dim, lr3_dim), interpolation=cv2.INTER_AREA)
    transform_ls_lr3 = from_bounds(xmin, ymin, xmax, ymax, lr3_dim, lr3_dim)

    with rasterio.open(os.path.join(folder + subfolder_10, nombre + "_RGB_LS_float_x3.tif"),
                        'w', driver='GTiff', height=lr3_dim, width=lr3_dim, count=3,
                        dtype='float32', crs='EPSG:4326', transform=transform_ls_lr3) as dst:
        for i in range(3):
            dst.write(rgb_ls_lr3[:, :, i], i + 1)

    rgb_ls_lr3_mod = to_uint8_rgb(rgb_ls_lr3)
    with rasterio.open(os.path.join(folder + subfolder_11, nombre + "_RGB_LS_mod_x3.tif"),
                        'w', driver='GTiff', height=lr3_dim, width=lr3_dim, count=3,
                        dtype='uint8', crs='EPSG:4326', transform=transform_ls_lr3) as dst:
        for i in range(3):
            dst.write(rgb_ls_lr3_mod[:, :, i], i + 1)

    rgb_ls_lr3_vis = to_uint8_vis(rgb_ls_lr3)
    with rasterio.open(os.path.join(folder + subfolder_12, nombre + "_RGB_LS_vis_x3.tif"),
                        'w', driver='GTiff', height=lr3_dim, width=lr3_dim, count=3,
                        dtype='uint8', crs='EPSG:4326', transform=transform_ls_lr3) as dst:
        for i in range(3):
            dst.write(rgb_ls_lr3_vis[:, :, i], i + 1)

    print("Guardado LR/3 Landsat")

    # ================= EXPORT FULL =================
    geemap.ee_export_image(
        images_dict["s2_full"],
        filename=os.path.join(folder + subfolder_3, nombre + "_FB_S2.tif"),
        region=roi, scale=10, file_per_band=False
    )
    geemap.ee_export_image(
        images_dict["landsat_full"],
        filename=os.path.join(folder + subfolder_4, nombre + "_FB_LS.tif"),
        region=roi, scale=30, file_per_band=False
    )
    print("Todo guardado correctamente")
