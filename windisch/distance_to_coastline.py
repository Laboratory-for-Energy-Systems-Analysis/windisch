import geopandas as gpd
from shapely.geometry import Point

from . import DATA_DIR

# Path to the GSHHS coastline shapefile (low resolution)
coastline_shapefile = DATA_DIR / 'gshhg-shp-2.3.7/GSHHS_shp/l/GSHHS_l_L1.shp'

def find_nearest_coastline(lat, lon):
    """
    Given a dataframe with coastline geometries and a lat/lon coordinate,
    returns the minimum distance to the nearest coastline in meters.
    """

    # Load coastline shapefile
    coastline_gdf = gpd.read_file(coastline_shapefile)

    # Create a point from the given lat/lon
    point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")  # WGS 84 (lat/lon)

    # Reproject both the coastline and the query point to a projected CRS
    projected_crs = "EPSG:3857"  # Web Mercator (meters) or use local UTM
    coastline_gdf = coastline_gdf.to_crs(projected_crs)
    point = point.to_crs(projected_crs)

    # Compute distances (in meters)
    coastline_gdf["distance"] = coastline_gdf["geometry"].distance(point.iloc[0])

    # Return the minimum distance (in meters)
    return coastline_gdf["distance"].min()
