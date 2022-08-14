LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H-%M")
# Run:
#  make help

# Requires:
#   docker 
#   docker-compose
#   aws cli

help:
	@echo "\nTARGETS:\n"
	@make -qpRr | egrep -e '^[a-z].*:$$' | sed -e 's~:~~g' | sort
	@echo ""
	
setup-model-registry:
	mkdir /tmp/mlopsdb
	mkdir /tmp/mlopsartifacts  
	mkdir /tmp/store   
	mkdir /tmp/serve   
	docker-compose -d -f docker-compose-model-registry.yml up --build --force-recreate
	

start-model-train-flow:
	aws s3 mb s3://capstone --endpoint-url=http://localhost:4566  
	aws s3 cp ../input_clean/credit_card_churn_clean.csv --endpoint-url=http://localhost:4566 s3://capstone/ID1/credit_card_churn_2022-08-07.csv
	prefect deployment create model_train_flow.py

setup-model-serve:
	docker-compose -d -f docker-compose-serve.yml  up --build --force-recreate

# start-serve-requests:
# 	pipenv run python send_data.py &>/dev/null &   

# test:
# 	pipenv run pytest tests/

# quality_checks:
# 	pipenv run pylint --recursive=y .

# build: quality_checks test
# 	docker build -t ${LOCAL_IMAGE_NAME} .

# integration_test: build
# 	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash run.sh

# publish: build integration_test
# 	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash publish.sh

# setup:
# 	pipenv install --dev
# 	pipenv run pre-commit install
