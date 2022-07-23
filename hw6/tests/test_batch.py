from datetime import datetime
from deepdiff import DeepDiff
import pandas as pd
import batch


def test_one():
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (1, 1, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime']
    df = pd.DataFrame(data, columns=columns)

    actual_result = batch.prepare_data(df, columns)
    # print(actual_result.to_dict)

    expected_columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime', 'duration']
    expected_data = [
        (-1, -1, 1609462920000000000, 1609463400000000000, 8.0),
        (1, 1, 1609462920000000000, 1609463400000000000, 8.0)
    ]
    expected_result = pd.DataFrame(expected_data, columns=expected_columns)
    # print(expected_result.to_dict)

    diff = DeepDiff(actual_result.to_dict, expected_result.to_dict, significant_digits=1)
    print(diff)
    assert diff == {}


def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)
