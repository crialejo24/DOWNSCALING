"""
gee_utils.py
Funciones de análisis de imágenes satelitales sobre Google Earth Engine (GEE):
- porcentaje de nubes (Sentinel-2 y Landsat)
- porcentaje de agua (NDWI)
- variabilidad espacial (coeficiente de variación RGB)

Requiere: earthengine-api (import ee) ya inicializado con tu propio
proyecto de GEE antes de llamar estas funciones.
"""

import ee


# =========================================================
# NUBES SENTINEL
# =========================================================
def cloud_percentage_s2(image, roi):
    """Calcula el % de píxeles con nubes en un ROI para una imagen Sentinel-2."""
    # ---------- SCL ----------
    scl = image.select('SCL')
    scl_clouds = (
        scl.eq(3)   # sombra
        .Or(scl.eq(8))
        .Or(scl.eq(9))
        .Or(scl.eq(10))
    )
    # .Or(scl.eq(7))  # Clouds Low Probability / Unclassified
    # .Or(scl.eq(11)) # Snow / Ice

    # ---------- QA60 ----------
    qa60 = image.select('QA60')
    qa_cloud = qa60.bitwiseAnd(1 << 10).neq(0)
    qa_cirrus = qa60.bitwiseAnd(1 << 11).neq(0)
    qa_clouds = qa_cloud.Or(qa_cirrus)

    # ---------- combinación ----------
    clouds = scl_clouds.Or(qa_clouds)
    stats = clouds.reduceRegion(
        reducer=ee.Reducer.sum().combine(
            reducer2=ee.Reducer.count(),
            sharedInputs=True
        ),
        geometry=roi,
        scale=20,
        maxPixels=1e9
    )
    cloud_pixels = ee.Number(stats.get('SCL_sum'))
    total_pixels = ee.Number(stats.get('SCL_count'))
    cloud_pct = cloud_pixels.divide(total_pixels).multiply(100)
    return image.set('CLOUD_PCT_ROI', cloud_pct)


# =========================================================
# AGUA SENTINEL (NDWI)
# =========================================================
def water_percentage_s2(image, roi, threshold=0.00):
    """Calcula el % de agua (NDWI > threshold) y el NDWI promedio en un ROI."""
    green = image.select('B3').multiply(1e-4)
    nir = image.select('B8').multiply(1e-4)
    ndwi = green.subtract(nir).divide(green.add(nir)).rename('NDWI')

    water_mask = ndwi.gt(threshold).rename('water')
    stats = water_mask.reduceRegion(
        reducer=ee.Reducer.sum().combine(
            reducer2=ee.Reducer.count(),
            sharedInputs=True
        ),
        geometry=roi,
        scale=10,
        maxPixels=1e9
    )
    water_pixels = ee.Number(stats.get('water_sum'))
    total_pixels = ee.Number(stats.get('water_count'))
    water_pct = water_pixels.divide(total_pixels).multiply(100)

    ndwi_mean = ndwi.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi,
        scale=10,
        maxPixels=1e9
    ).get('NDWI')

    return image.set({
        'WATER_PCT_ROI': water_pct,
        'NDWI_MEAN_ROI': ndwi_mean
    })


# =========================================================
# NUBES LANDSAT
# =========================================================
def cloud_percentage_landsat(image, roi):
    """Calcula el % de píxeles con nubes/sombra en un ROI para una imagen Landsat."""
    qa = image.select('QA_PIXEL')
    dilated = qa.bitwiseAnd(1 << 1).neq(0)
    cirrus = qa.bitwiseAnd(1 << 2).neq(0)
    cloud = qa.bitwiseAnd(1 << 3).neq(0)
    shadow = qa.bitwiseAnd(1 << 4).neq(0)
    # snow = qa.bitwiseAnd(1 << 5).neq(0)

    clouds = dilated.Or(cirrus).Or(cloud).Or(shadow)  # .Or(snow)
    stats = clouds.reduceRegion(
        reducer=ee.Reducer.sum().combine(
            reducer2=ee.Reducer.count(),
            sharedInputs=True
        ),
        geometry=roi,
        scale=30,
        maxPixels=1e9
    )
    cloud_pixels = ee.Number(stats.get('QA_PIXEL_sum'))
    total_pixels = ee.Number(stats.get('QA_PIXEL_count'))
    cloud_pct = cloud_pixels.divide(total_pixels).multiply(100)
    return image.set('CLOUD_PCT_ROI', cloud_pct)


# =========================================================
# VARIABILIDAD SENTINEL (CV RGB %)
# =========================================================
def variability_s2(image, roi):
    """Calcula el coeficiente de variación (%) de las bandas RGB en un ROI."""
    rgb = image.select(['B4', 'B3', 'B2']).multiply(1e-4)
    stats = rgb.reduceRegion(
        reducer=ee.Reducer.mean().combine(
            reducer2=ee.Reducer.stdDev(),
            sharedInputs=True
        ),
        geometry=roi,
        scale=10,
        maxPixels=1e9
    )
    mean_r = ee.Number(stats.get('B4_mean'))
    mean_g = ee.Number(stats.get('B3_mean'))
    mean_b = ee.Number(stats.get('B2_mean'))
    std_r = ee.Number(stats.get('B4_stdDev'))
    std_g = ee.Number(stats.get('B3_stdDev'))
    std_b = ee.Number(stats.get('B2_stdDev'))

    mean_rgb = mean_r.add(mean_g).add(mean_b).divide(3)
    std_rgb = std_r.add(std_g).add(std_b).divide(3)
    cv_rgb = ee.Algorithms.If(
        mean_rgb.neq(0),
        std_rgb.divide(mean_rgb).multiply(100),
        ee.Number(0)
    )
    return image.set('STD_RGB_ROI', cv_rgb)
