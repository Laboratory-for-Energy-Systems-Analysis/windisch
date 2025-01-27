import netCDF4
import numpy as np


def get_sea_depth(data, latitude, longitude):
    """
    Fetch the sea depth at a specific latitude and longitude from a GEBCO NetCDF file.

    :param data: The NetCDF file object.
    :param latitude: The latitude in decimal degrees.
    :param longitude: The longitude in decimal degrees.

    """

    # Read the latitude and longitude arrays from the dataset
    latitudes = data.variables["lat"][:]
    longitudes = data.variables["lon"][:]

    # Handle longitudes that might be 0-360 instead of -180 to 180
    if longitude < 0:
        longitude += 360

    # Find the nearest indices to the requested latitude and longitude
    lat_idx = np.abs(latitudes - latitude).argmin()
    lon_idx = np.abs(longitudes - longitude).argmin()

    # Fetch the depth value at the nearest indices
    depth = data.variables["elevation"][lat_idx, lon_idx]

    # Close the NetCDF file
    data.close()

    return depth
