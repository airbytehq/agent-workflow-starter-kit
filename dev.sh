#! /usr/bin/env sh

set -e
cd "$(dirname "$0")"

# load env & overrides
set -o allexport
source ./.env

cd app

# Inspired by https://github.com/tiangolo/uvicorn-gunicorn-docker/blob/master/docker-images/start.sh
PRE_START_PATH=${PRE_START_PATH:-/app/prestart.sh}
if [ -f $PRE_START_PATH ] ; then
    echo "Running script $PRE_START_PATH"
    . "$PRE_START_PATH"
fi

poetry run download_embedding_model
poetry run uvicorn main:application --reload --port 8080
