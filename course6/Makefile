LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H-%M")
LOCAL_IMAGE_NAME:=stream-model-duration:${LOCAL_TAG}

# Requires:
#   conda activate course5
#   and pipenv installed

start:
	echo Stating pipeline for image: ${LOCAL_IMAGE_NAME}

test:
	pipenv run pytest tests/

quality_checks:
	pipenv run pylint --recursive=y .

build: quality_checks test
	docker build -t ${LOCAL_IMAGE_NAME} .

integration_test: build
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash run.sh

publish: build integration_test
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash publish.sh

setup:
	pipenv install --dev
	pipenv run pre-commit install
