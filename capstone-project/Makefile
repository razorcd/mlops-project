LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H-%M")
# Run:
#  make help

# Requires:
#   make
#   docker 
#   docker-compose
#   aws cli
#   pip3b install prefect==2.0b5

help:
	@echo "\nTARGETS:\n"
	@make -qpRr | egrep -e '^[a-z].*:' | sed -e 's~:~~g' | sort
	@echo ""
	
setup-model-registry:
	mkdir -p /tmp/mlopsdb
	mkdir -p /tmp/mlopsartifacts  
	mkdir -p /tmp/store   
	mkdir -p /tmp/serve   
	docker compose --env-file=.env -f docker-compose-model-registry.yml up --build --force-recreate
	
start-model-train-flow:
	aws s3 mb s3://capstone --endpoint-url=http://localhost:4566 || true
	aws s3 cp input_clean/credit_card_churn_clean.csv --endpoint-url=http://localhost:4566 s3://capstone/ID1/credit_card_churn_2022-08-07.csv
	docker cp ./orchestration/model_train_flow.py prefect_agent_1:model_train_flow.py
	docker exec prefect_agent_1 prefect deployment create model_train_flow.py

setup-model-serve:
	chmod 777 monitoring/
	chmod 777 monitoring/**/*
	docker compose --env-file=.env -f docker-compose-serve.yml  up --build --force-recreate

stop-serve:	
	docker rm -f serve
	docker rm -f send_data

start-serve:
	docker compose --env-file=.env -f docker-compose-serve.yml up -d



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
