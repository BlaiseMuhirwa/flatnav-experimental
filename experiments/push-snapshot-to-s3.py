import boto3
import os
import sys
import logging
from datetime import datetime
from typing import Optional
import time


logging.basicConfig(level=logging.INFO)


logger = logging.getLogger(__name__)


def upload_file_to_s3(
    s3_client: boto3.client, file_name: str, bucket_name: str, bucket_prefix: str
):
    """
    Uploads a file to an S3 bucket.
    :param s3_client: The S3 client to use.
    :param file_name: The file to upload.
    :param bucket_name: The name of the bucket to upload to.
    :param bucket_prefix: The prefix to add to the file in the bucket.
    """
    if not bucket_prefix.endswith("/"):
        bucket_prefix += "/"

    object_name = f"{bucket_prefix}{os.path.basename(file_name)}"
    logger.info(f"Attempting to upload file {file_name} to {object_name}")

    try:
        s3_client.upload_file(file_name, bucket_name, object_name)

    except Exception:
        logger.error(
            f"Failed to upload file {file_name} to {object_name}", exc_info=True
        )
        raise


def delete_old_indices(directory: str, file_extension: str) -> None:
    for file in os.listdir(directory):
        if file.startswith("index_") and file.endswith(file_extension):
            logging.info(f"Deleting old index file {file}")
            os.remove(os.path.join(directory, file))


def find_most_recent_file(directory: str, file_extension: str) -> str:
    """
    Finds the most recent file in a directory with a given file extension.
    :param directory: The directory to search.
    :param file_extension: The file extension to search for.
    :return: The path to the most recent file.
    """
    latest_time = datetime.min
    latest_file = None
    time_format = "%a%b%d%H:%M:%S%Y"

    for file in os.listdir(directory):
        if file.endswith(file_extension) and file.startswith("index_"):
            # Extract the datetime part by removing prefix and suffix
            datetime_part = file[len("index_") : -len(file_extension)]
            try:
                filetime = datetime.strptime(datetime_part, time_format)
                if filetime > latest_time:
                    latest_time = filetime
                    latest_file = os.path.join(directory, file)
            except ValueError:
                logger.error(f"Error parsing the file {file}", exc_info=True)

    return latest_file


def run():
    """
    Uploads the most recent HNSW index snapshot to an S3 bucket specified by the
    AWS_S3_BUCKET_NAME and MAKE_TARGET environment variables.

    Indexes are expected to be in the current working directory and have the
    extension specified by the FILE_EXTENSION environment variable.
    For instance, one of them might look like index_Sun May 02 20:00:01 2021.hnsw
    """
    bucket_name = os.environ.get("AWS_S3_BUCKET_NAME")
    file_extension = os.environ.get("FILE_EXTENSION", ".hnsw")
    bucket_prefix = os.environ.get("MAKE_TARGET")

    if not (bucket_name and bucket_prefix):
        raise ValueError(
            "AWS_S3_BUCKET_NAME and BUCKET_PREFIX environment variables must be set"
        )

    s3_client = boto3.client("s3")

    while True:
        latest_file = find_most_recent_file(
            directory=os.getcwd(), file_extension=file_extension
        )

        if latest_file:
            upload_file_to_s3(
                s3_client=s3_client,
                file_name=latest_file,
                bucket_name=bucket_name,
                bucket_prefix=bucket_prefix,
            )

            delete_old_indices(directory=os.getcwd(), file_extension=file_extension)
        else:
            logger.info(
                f"No file with extension {file_extension} found in {os.getcwd()}"
            )

        time.sleep(30)


if __name__ == "__main__":
    logger.info("Running push-snapshot-to-s3.py")

    DISABLE_PUSH_TO_S3 = bool(int(os.environ.get("DISABLE_PUSH_TO_S3", "0")))

    if DISABLE_PUSH_TO_S3:
        logger.info(
            "DISABLE_PUSH_TO_S3 environment variable is set. No indexes will be pushed."
        )
        while True:
            # Delete any indexes that have accumulated
            delete_old_indices(
                directory=os.getcwd(),
                file_extension=os.environ.get("FILE_EXTENSION", ".hnsw"),
            )
            time.sleep(30)

    run()
