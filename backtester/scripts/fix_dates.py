import argparse
import csv
import os
from datetime import datetime, timedelta

import pandas as pd


def infer_close_time(row) -> str:
    open_time = pd.to_datetime(row[0]).tz_localize('America/New_York').to_pydatetime()
    close_time = open_time + timedelta(seconds=59)
    return close_time.isoformat(sep=' ')


def localize_open_time(row) -> str:
    open_time = pd.to_datetime(row[0]).tz_localize('America/New_York').to_pydatetime()
    return open_time.isoformat(sep=' ')


def get_output_filename(input_filename: str) -> str:
    without_ext = input_filename.split('.')[0]
    return f"{without_ext}_fixed.csv"


def main():
    """
    Writes content of an input file, adding close time column,
    inferred from the open time. Adds time zone info (America/New_York)
    to open time and close time columns.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="filename to fix, searched in data/ directory",
                        type=str, required=True)
    # Parse filename
    args = parser.parse_args()
    filename = args.filename
	# Evaluate lookup dirs
    root_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(root_dir, 'data')

    input_file = os.path.join(data_dir, filename)
    output_file = os.path.join(data_dir, get_output_filename(filename))

    with open(output_file, 'w', newline='') as output:
        with open(input_file, 'r', newline='') as input_:
            reader = csv.reader(input_, delimiter=',')
            writer = csv.writer(output, delimiter=',')
            # Skip the headers
            next(reader)
            # Write headers
            writer.writerow([
                'open_time', 
                'open', 
                'high', 
                'low', 
                'close',
                'volume',
                'close_time',
            ])
            # Write content
            for row in reader:
                writer.writerow([
                    localize_open_time(row),
                    row[1], # open
                    row[2], # high
                    row[3], # low
                    row[4], # close
                    row[5], # volume
                    infer_close_time(row)
                ])


if __name__ == '__main__':
	main()
