from typing import Dict, List, Tuple
import os
from tsx.datasets.utils import download_and_unzip
import pandas as pd
import numpy as np


def load_m5(include_exo: bool, cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] or pd.DataFrame:
        """Downloads and loads M5 data.

        Parameters
        ----------
        include_exo: bool
            If `True` includes additional data (X_df and S_df). Otherwise, only returns Y_df.
        cache: bool
            If `True` saves and loads.

        Returns
        -------
        Y_df: pd.DataFrame
            Target time series with columns ['unique_id', 'ds', 'y'].
        X_df: pd.DataFrame
            Exogenous time series with columns ['unique_id', 'ds', 'y'].
        S_df: pd.DataFrame
            Static exogenous variables with columns ['unique_id', 'ds'].
            and static variables.
        """
        path = os.path.join(os.path.dirname(__file__), "data", "M5")
        file_cache = f'{path}/m5.p'

        if cache and os.path.exists(file_cache):
            Y_df, X_df, S_df = pd.read_pickle(file_cache)
            if include_exo:
                return Y_df, X_df, S_df
            else:
                return Y_df

        download_m5()
        # Calendar data
        cal_dtypes = {
            'wm_yr_wk': np.uint16,
            'event_name_1': 'category',
            'event_type_1': 'category',
            'event_name_2': 'category',
            'event_type_2': 'category',
            'snap_CA': np.uint8,
            'snap_TX': np.uint8,
            'snap_WI': np.uint8,
        }
        cal = pd.read_csv(f'{path}/calendar.csv',
                          dtype=cal_dtypes,
                          usecols=list(cal_dtypes.keys()) + ['date'],
                          parse_dates=['date'])
        cal['d'] = np.arange(cal.shape[0]) + 1
        cal['d'] = 'd_' + cal['d'].astype('str')
        cal['d'] = cal['d'].astype('category')

        event_cols = [k for k in cal_dtypes if k.startswith('event')]
        for col in event_cols:
            cal[col] = cal[col].cat.add_categories('nan').fillna('nan')

        # Prices
        prices_dtypes = {
            'store_id': 'category',
            'item_id': 'category',
            'wm_yr_wk': np.uint16,
            'sell_price': np.float32
        }

        prices = pd.read_csv(f'{path}/sell_prices.csv',
                             dtype=prices_dtypes)

        # Sales
        sales_dtypes = {
            'item_id': prices.item_id.dtype,
            'dept_id': 'category',
            'cat_id': 'category',
            'store_id': 'category',
            'state_id': 'category',
            **{f'd_{i + 1}': np.float32 for i in range(1969)}
        }
        # Reading train and test sets
        sales_train = pd.read_csv(f'{path}/sales_train_evaluation.csv', dtype=sales_dtypes)
        sales_test = pd.read_csv(f'{path}/sales_test_evaluation.csv', dtype=sales_dtypes)
        sales = sales_train.merge(sales_test, how='left',
                                  on=['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])
        sales['id'] = sales['item_id'].astype(str) + '_' + sales['store_id'].astype(str)
        sales['id'] = sales['id'].astype('category')
        # Long format
        long = sales.melt(id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
                          var_name='d', value_name='y')
        # paste dates
        long['d'] = long['d'].astype(cal.d.dtype)
        long = long.merge(cal, on=['d'])

        # remove leading zeros from series
        long = long.sort_values(['id', 'date'])
        without_leading_zeros = long['y'].gt(0).groupby(long['id']).transform('cummax')
        long = long[without_leading_zeros]

        # prices
        long = long.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'])
        long = long.drop(columns=['d', 'wm_yr_wk'])

        long = long.rename(columns={'id': 'unique_id', 'date': 'ds'})
        Y_df = long[['unique_id', 'ds', 'y']]
        cats = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        S_df = long[['unique_id'] + cats].groupby('unique_id', observed=True).head(1)
        x_cols = long.columns.drop(['y'] + cats)
        X_df = long[x_cols]

        if cache:
            pd.to_pickle((Y_df, X_df, S_df), file_cache)

        if include_exo:
            return Y_df, X_df, S_df
        else:
            return Y_df


def download_m5() -> str:
    """
    Downloads the m5 dataset
    """
    name = "M5"
    path = os.path.join(os.path.dirname(__file__), "data", name)

    if not os.path.exists(path):
        url = 'https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip'
        download_and_unzip(url, name)
        print('downloaded', path)
    else:
        print(f"{path} already exists")
    return path


if __name__ == "__main__":
    x, y, z = load_m5(include_exo=True, cache=True)
    print(y["event_name_1"].unique())
