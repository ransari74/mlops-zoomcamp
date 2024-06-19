import argparse
import os
import pickle

import pandas as pd

with open("model.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)


# Function to upload DataFrame as Parquet to S3
def upload_parquet_to_s3(df: pd.DataFrame, s3_path):
    try:
        # Write the DataFrame directly to the S3 path as a Parquet file
        df.to_parquet(
            s3_path,
            index=False,
            storage_options={
                "key": os.environ.get("AWS_ACCESS_KEY_ID", ""),
                "secret": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
            },
        )
        print(f"File uploaded successfully to {s3_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


categorical = ["PULocationID", "DOLocationID"]


def read_data(filename):
    df = pd.read_parquet(filename)

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a test.")

    parser.add_argument("year", type=str, help="year in yyyy format")
    parser.add_argument("month", type=str, help="month in mm format")
    parser.add_argument(
        "--output_file_name",
        type=str,
        help="output_file_name on s3",
        default=f"prediction_{pd.Timestamp.now().to_pydatetime()}.parquet",
    )

    args = parser.parse_args()

    # S3 bucket and file path
    bucket_name = os.environ.get("BUCKET_NAME", "")
    file_key = args.output_file_name
    s3_path = f"s3://{bucket_name}/{file_key}"
    df = read_data(
        f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{args.year}-{args.month}.parquet"
    )
    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    print("std: ", y_pred.std(), "mean: ", y_pred.mean())

    df["ride_id"] = f"{args.year}/{args.month}_" + df.index.astype("str")
    df["prediction_value"] = y_pred
    # Upload the DataFrame as a Parquet file to S3
    upload_parquet_to_s3(df[["ride_id", "prediction_value"]], s3_path)
