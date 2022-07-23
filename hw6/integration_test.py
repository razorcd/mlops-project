#!/usr/bin/env python
# coding: utf-8

from datetime import datetime
import pandas as pd


def test_integration_one():
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (1, 1, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime']
    df_input = pd.DataFrame(data, columns=columns)

    storage_options = {
        'client_kwargs': {'endpoint_url': 'http://localhost:4566'}
    }

    input_file = 's3://nyc-duration/ID1/testdata.parquet'

    df_input.to_parquet(
        input_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=storage_options
    )

def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)


if __name__ == "__main__":
    test_integration_one()

