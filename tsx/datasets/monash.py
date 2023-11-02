from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
from datetime import datetime
from distutils.util import strtobool
import pandas as pd
import os
import torch

from tsx.datasets.utils import download_and_unzip

univariate_datasets = [
    'm1_yearly',
    'm1_quarterly',
    'm1_monthly',
    'm3_yearly',
    'm3_quarterly',
    'm3_monthly',
    'm3_other',
    'm4_yearly',
    'm4_quarterly',
    'm4_monthly',
    'm4_weekly',
    'm4_daily',
    'm4_hourly',
    'tourism_yearly',
    'tourism_quarterly',
    'tourism_monthly',
    'cif_2016',
    #'london_smart_meters_nomissing',
    'australian_electricity_demand',
    #'wind_farms_nomissing',
    'dominick',
    'bitcoin_nomissing',
    'pedestrian_counts',
    'vehicle_trips_nomissing',
    'kdd_cup_nomissing',
    'weather',
    'sunspot_nomissing',
    'saugene_river_flow',
    'us_births',
    #'solar_power',
    #'wind_power',
]

multivariate_datasets = [
    'nn5_daily_nomissing',
    'nn5_weekly',
    'web_traffic_daily_nomissing',
    'web_traffic_weekly',
    'solar_10_minutes',
    'electricity_hourly',
    'electricity_weekly',
    'car_parts_nomissing',
    'san_fracisco_traffic_hourly',
    'san_fracisco_traffic_weekly',
    'ride_share_nomissing',
    'hospital',
    'fred_md',
    'covid_deaths',
    'temperature_rain_nomissing',
]

# From Cerqueira et al. 2023 "Model Selection for Time Series Forecasting An Empirical Analysis of Multiple Estimators"
def load_m4_daily_bench(min_size=500, return_horizon=False):
    data, horizons = load_monash('m4_daily', return_horizon=True)
    horizons = np.array(horizons)
    data = data['series_value']

    indices = np.where([len(ts) >= min_size for ts in data])[0]

    if return_horizon:
        return [ts.to_numpy() for ts in data.iloc(indices)], horizons[indices]
    return [ts.to_numpy() for ts in data.iloc(indices)]

def possible_datasets():
    """ Returns list of possible dataset names
    """
    return list(get_links_dict().keys())


def load_monash(dataset: str, return_pytorch: bool = False, return_numpy: bool = False, return_horizon: bool = False):
    """ Loads datasets from Monash Time Series Forecasting Repository.

    Args:
        dataset: Name of the dataset to be downloaded. Consists of the name of the dataset as well as the "version" of the dataset separated by an underscore.
        return_horizon: Datasets have a specific forecast horizon. True if they should be returned as well.
        return_pytorch: Returns dataset as a PyTorch tensor. Throws error if not possible.
        return_numpy: Returns dataset as a numpy array. Throws error if not possible.
    """
    path = download(dataset)
    files = os.listdir(path)
    files = [file for file in files if file.endswith(".tsf")]

    if len(files) < 1:
        raise Exception("no .tsf file found!")
    if len(files) > 1:
        raise Exception("multiple .tsf files found!")

    frame, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(os.path.join(path, files[0]))
    series_value = frame["series_value"]

    # Set forecast horizon  manually, based on https://arxiv.org/pdf/2202.08485.pdf
    if dataset == 'australian_electricity_demand':
        forecast_horizon = 48
    elif dataset == 'dominick':
        forecast_horizon = 8
    elif dataset == 'bitcoin_nomissing' or dataset == 'bitcoin_missing':
        forecast_horizon = 30
    elif dataset == 'pedestrian_counts':
        forecast_horizon = 48
    elif dataset == 'vehicle_trips_missing' or dataset == 'vehicle_trips_nomissing':
        forecast_horizon = 30
    elif dataset == 'kdd_cup_missing' or dataset == 'kdd_cup_nomissing':
        forecast_horizon = 48
    elif dataset == 'weather':
        forecast_horizon = 30

    if forecast_horizon is None:
        if 'horizon' in frame.columns:
            forecast_horizon = frame['horizon'].tolist()
    else:
        forecast_horizon = [forecast_horizon for _ in range(len(series_value))]

    if return_numpy or return_pytorch:
        if contain_equal_length:
            array = series_value.to_numpy()
            for i, p_array in enumerate(array):
                array[i] = p_array.to_numpy()
            array = np.stack(array).astype("float64")
            if dataset in multivariate_datasets:
                array = array.T
            array = array.squeeze()
            if return_pytorch:
                array = torch.from_numpy(array)

            if return_horizon:
                return array, forecast_horizon
            else:
                return array

    if return_horizon:
        return frame, forecast_horizon
    return frame


def get_links_dict() -> Dict[str, str]:
    """ Get dictionary with dataset names as keys and corresponding download urls as values
    """
    return {
        "m1_yearly": "https://zenodo.org/record/4656193/files/m1_yearly_dataset.zip",
        "m1_quarterly": "https://zenodo.org/record/4656154/files/m1_quarterly_dataset.zip",
        "m1_monthly": "https://zenodo.org/record/4656159/files/m1_monthly_dataset.zip",
        "m3_yearly": "https://zenodo.org/record/4656222/files/m3_yearly_dataset.zip",
        "m3_quarterly": "https://zenodo.org/record/4656262/files/m3_quarterly_dataset.zip",
        "m3_monthly": "https://zenodo.org/record/4656298/files/m3_monthly_dataset.zip",
        "m3_other": "https://zenodo.org/record/4656335/files/m3_other_dataset.zip",
        "m4_yearly": "https://zenodo.org/record/4656379/files/m4_yearly_dataset.zip",
        "m4_quarterly": "https://zenodo.org/record/4656410/files/m4_quarterly_dataset.zip",
        "m4_monthly": "https://zenodo.org/record/4656480/files/m4_monthly_dataset.zip",
        "m4_weekly": "https://zenodo.org/record/4656522/files/m4_weekly_dataset.zip",
        "m4_daily": "https://zenodo.org/record/4656548/files/m4_daily_dataset.zip",
        "m4_hourly": "https://zenodo.org/record/4656589/files/m4_hourly_dataset.zip",
        "tourism_yearly": "https://zenodo.org/record/4656103/files/tourism_yearly_dataset.zip",
        "tourism_quarterly": "https://zenodo.org/record/4656093/files/tourism_quarterly_dataset.zip",
        "tourism_monthly": "https://zenodo.org/record/4656096/files/tourism_monthly_dataset.zip",
        "cif_2016": "https://zenodo.org/record/4656042/files/cif_2016_dataset.zip",
        "london_smart_meters_missing": "https://zenodo.org/record/4656072/files/london_smart_meters_dataset_with_missing_values.zip",
        "london_smart_meters_nomissing": "https://zenodo.org/record/4656091/files/london_smart_meters_dataset_without_missing_values.zip",
        "australian_electricity_demand": "https://zenodo.org/record/4659727/files/australian_electricity_demand_dataset.zip",
        "wind_farms_missing": "https://zenodo.org/record/4654909/files/wind_farms_minutely_dataset_with_missing_values.zip",
        "wind_farms_nomissing": "https://zenodo.org/record/4654858/files/wind_farms_minutely_dataset_without_missing_values.zip",
        "dominick": "https://zenodo.org/record/4654802/files/dominick_dataset.zip",
        "bitcoin_missing": "https://zenodo.org/record/5121965/files/bitcoin_dataset_with_missing_values.zip",
        "bitcoin_nomissing": "https://zenodo.org/record/5122101/files/bitcoin_dataset_without_missing_values.zip",
        "pedestrian_counts": "https://zenodo.org/record/4656626/files/pedestrian_counts_dataset.zip",
        "vehicle_trips_missing": "https://zenodo.org/record/5122535/files/vehicle_trips_dataset_with_missing_values.zip",
        "vehicle_trips_nomissing": "https://zenodo.org/record/5122537/files/vehicle_trips_dataset_without_missing_values.zip",
        "kdd_cup_missing": "https://zenodo.org/record/4656719/files/kdd_cup_2018_dataset_with_missing_values.zip",
        "kdd_cup_nomissing": "https://zenodo.org/record/4656756/files/kdd_cup_2018_dataset_without_missing_values.zip",
        "weather": "https://zenodo.org/record/4654822/files/weather_dataset.zip",
        "nn5_daily_missing": "https://zenodo.org/record/4656110/files/nn5_daily_dataset_with_missing_values.zip",
        "nn5_daily_nomissing": "https://zenodo.org/record/4656117/files/nn5_daily_dataset_without_missing_values.zip",
        "nn5_weekly": "https://zenodo.org/record/4656125/files/nn5_weekly_dataset.zip",
        "web_traffic_daily_missing": "https://zenodo.org/record/4656080/files/kaggle_web_traffic_dataset_with_missing_values.zip",
        "web_traffic_daily_nomissing": "https://zenodo.org/record/4656075/files/kaggle_web_traffic_dataset_without_missing_values.zip",
        "web_traffic_weekly": "https://zenodo.org/record/4656664/files/kaggle_web_traffic_weekly_dataset.zip",
        "solar_10_minutes": "https://zenodo.org/record/4656144/files/solar_10_minutes_dataset.zip",
        "solar_weekly": "https://zenodo.org/record/4656151/files/solar_weekly_dataset.zip",
        "electricity_hourly": "https://zenodo.org/record/4656140/files/electricity_hourly_dataset.zip",
        "electricity_weekly": "https://zenodo.org/record/4656141/files/electricity_weekly_dataset.zip",
        "car_parts_missing": "https://zenodo.org/record/4656022/files/car_parts_dataset_with_missing_values.zip",
        "car_parts_nomissing": "https://zenodo.org/record/4656021/files/car_parts_dataset_without_missing_values.zip",
        "fred_md": "https://zenodo.org/record/4654833/files/fred_md_dataset.zip",
        "san_fracisco_traffic_hourly": "https://zenodo.org/record/4656132/files/traffic_hourly_dataset.zip",
        "san_fracisco_traffic_weekly": "https://zenodo.org/record/4656135/files/traffic_weekly_dataset.zip",
        "ride_share_missing": "https://zenodo.org/record/5122114/files/rideshare_dataset_with_missing_values.zip",
        "ride_share_nomissing": "https://zenodo.org/record/5122232/files/rideshare_dataset_without_missing_values.zip",
        "hospital": "https://zenodo.org/record/4656014/files/hospital_dataset.zip",
        "covid_deaths": "https://zenodo.org/record/4656009/files/covid_deaths_dataset.zip",
        "temperature_rain_missing": "https://zenodo.org/record/5129073/files/temperature_rain_dataset_with_missing_values.zip",
        "temperature_rain_nomissing": "https://zenodo.org/record/5129091/files/temperature_rain_dataset_without_missing_values.zip",
        "sunspot_missing": "https://zenodo.org/record/4654773/files/sunspot_dataset_with_missing_values.zip",
        "sunspot_nomissing": "https://zenodo.org/record/4654722/files/sunspot_dataset_without_missing_values.zip",
        "saugene_river_flow": "https://zenodo.org/record/4656058/files/saugeenday_dataset.zip",
        "us_births": "https://zenodo.org/record/4656049/files/us_births_dataset.zip",
        "solar_power": "https://zenodo.org/record/4656027/files/solar_4_seconds_dataset.zip",
        "wind_power": "https://zenodo.org/record/4656032/files/wind_4_seconds_dataset.zip"
    }


def download(name: str) -> str:
    """ Downloads a given monash dataset

    Args:
        name: name of the dataset
    """
    path = os.path.join(os.path.dirname(__file__), "data", name)

    if not os.path.exists(path):
        links = get_links_dict()
        try:
            url = links[name]
        except KeyError:
            raise KeyError(f"no dataset with the name \"{name}\" found!")

        path = download_and_unzip(url, name)
        print('downloaded', path)
    return path


def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
) -> Tuple:
    """ Converts the contents in a .tsf file into a dataframe and returns it along with other meta-data of the dataset: frequency, horizon, whether the dataset contains missing values and whether the series have equal lengths
    Args:
        full_file_path_and_name: complete .tsf file path
        replace_missing_vals_with: a term to indicate the missing values in series in the returning dataframe
        value_column_name: Any name that is preferred to have as the name of the column containing series values in the returning dataframe
    """
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                    len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                    len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                                numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )


if __name__ == "__main__":
    name = "nn5_daily_missing"
    np_array = load_monash(name, return_numpy=True)
    print(np_array.shape)
    print(np_array.dtype)

    d = load_monash(name, return_pytorch=True)
    print(d.shape)
