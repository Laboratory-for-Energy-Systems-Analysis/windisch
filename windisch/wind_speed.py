import os
import tempfile

import numpy as np
import requests
import xarray as xr
from dotenv import load_dotenv
from requests.exceptions import Timeout
import pandas as pd

load_dotenv(dotenv_path="../.env")

API_NEWA_TIME_SERIES = os.getenv("API_NEWA_TS")
API_NEWA = os.getenv("API_NEWA")


def fetch_wind_speed(data):

    # fill-in NaNs with nearest values
    data = data.ffill(dim='time').bfill(dim='time')

    # Step 1: Create a time series for 8760 hours
    time_range = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h")

    # Extract corresponding month and hour values for the entire time range
    months = time_range.month
    hours = time_range.hour

    # Step 2: Interpolate the data along the 'month' and 'hour' dimensions
    # Align with the `time_range` sequence
    # Step 2: Interpolate without fill_value and handle NaNs separately
    data = data.interp(
        month=("time", months),
        hour=("time", hours),
        method="linear"
    )

    # Step 3: Fill NaNs (optional)
    # Fill any remaining NaNs due to edge cases
    data = data.ffill(dim="time").bfill(dim="time")
    # Assign the time coordinate to the resulting DataArray
    data = data.assign_coords(time=("time", time_range))
    # Drop the now-unused 'month' and 'hour' dimensions if they remain
    data = data.drop_vars(["month", "hour"], errors="ignore")
    # remove all coordinates except "time"
    data = data.drop_vars([c for c in data.coords if c not in ("time", "height")])

    return data

def fetch_terrain_variables(latitude: float, longitude: float, fetch_wind_data: bool) -> xr.DataArray:
    """
    Fetch wind speed data for a specific location using the NEWA API.

    :param latitude: Latitude of the location.
    :type latitude: float
    :param longitude: Longitude of the location.
    :type longitude: float
    :param heights: Heights at which to interpolate the wind speed data.
    :type heights: xr.DataArray
    :return: Simplified xarray.Dataset containing wind direction (WD10) and wind speed (WS10).
    :rtype: xr.DataArray
    """

    if not API_NEWA_TIME_SERIES:
        raise EnvironmentError("API_NEWA environment variable is not set.")
    if not API_NEWA:
        raise EnvironmentError("API_NEWA environment variable is not set.")

    if fetch_wind_data is True:
        # Construct the API URL
        url = API_NEWA_TIME_SERIES.replace("longitude=X", f"longitude={longitude}").replace(
            "latitude=X", f"latitude={latitude}"
        )

    else:
        print("Fetching terrain data only.")
        url = API_NEWA.replace("longitude=X", f"longitude={longitude}").replace(
            "latitude=X", f"latitude={latitude}"
        )

    attempts = 0
    max_attempts = 10

    while attempts < max_attempts:
        try:
            # Send the request
            response = requests.get(url, timeout=25)

            # Check if the response is successful
            if response.status_code == 200:
                size_kb = len(response.content) / 1024  # Calculate response size in KB
                print(
                    f"Downloaded {size_kb:.2f} kB for location ({latitude}, {longitude})"
                )

                # Write content to a temporary file
                with tempfile.NamedTemporaryFile(
                    suffix=".nc", delete=False
                ) as tmp_file:
                    tmp_file.write(response.content)
                    tmp_file_path = tmp_file.name

                # Open the dataset
                ds = xr.open_dataset(tmp_file_path)

                if fetch_wind_data is False:
                    # rename landmask to LANDMASK, tke50_mean to TAKE, rho_mean to RHO
                    ds = ds.rename_vars({"landmask": "LANDMASK", "tke50_mean": "TKE", "rho_mean": "RHO"})

                    # remove all coordinates except "time"
                    ds = ds.drop_vars([c for c in ds.coords if c != "time"])
                    return ds

                # the dataset has data points every 30 min
                # resample it to every hour
                ds = ds.resample(time="1h").mean()
                # average over the years to get 8760 representative hours
                # Convert time to 'hour of the year' (0 to 8759)
                hour_of_year = (ds['time'].dt.dayofyear - 1) * 24 + ds['time'].dt.hour

                # Group by hour_of_year and calculate the mean
                representative_hours = ds.groupby(hour_of_year).mean(dim='time')

                # Add a time coordinate for the 8760 hours (optional, for clarity)
                representative_hours['time'] = xr.cftime_range(start="2000-01-01", periods=8760, freq="h")
                representative_hours = representative_hours.set_coords('time')
                representative_hours = representative_hours.fillna(0)

                # remove "time" coordinate
                representative_hours = representative_hours.drop_vars("time")
                # and rename "group" to time
                representative_hours = representative_hours.rename(group="time")

                return representative_hours

            else:
                raise Exception(
                    f"Failed to fetch data. HTTP Status: {response.status_code}"
                )

        except (Timeout, Exception) as e:
            attempts += 1
            print(f"Error: {e}. Retrying ({attempts}/{max_attempts})...")

            if attempts >= max_attempts:
                raise Exception(
                    f"Failed to fetch data for location ({latitude}, {longitude}) after {max_attempts} attempts."
                )


