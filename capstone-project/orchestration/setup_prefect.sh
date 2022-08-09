#!/usr/bin/env bash

set -e

echo "Setting up Prefect."

echo "Create working queue."
pipenv run prefect work-queue create work_queue_3

echo "Creating storage."
printf "1\n/tmp/store\n\nstoreName\ny\n" | prefect storage create