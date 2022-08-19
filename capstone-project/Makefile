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

run_tests:
	pipenv run pytest server/tests	

reset_all:
	# docker compose -f docker-compose-serve.yml down
	docker compose -f docker-compose-model-registry.yml down
	rm -rf /tmp/mlopsdb
	rm -rf /tmp/mlopsartifacts  
	rm -rf /tmp/store   
	rm -rf /tmp/serve   
	rm -rf /tmp/mlreports
	rm .env

# make setup-model-registry env=local
setup-model-registry:
	mkdir -p /tmp/mlopsdb
	mkdir -p /tmp/mlopsartifacts  
	mkdir -p /tmp/store   
	mkdir -p /tmp/serve   
	mkdir -p /tmp/mlreports
	cp .env.$(env) .env
	docker compose -f docker-compose-model-registry.yml up --build --force-recreate
	

init_aws:
	aws s3 mb s3://capstone --endpoint-url=http://localhost:4566 || true
	aws s3 cp input_clean/credit_card_churn_clean.csv --endpoint-url=http://localhost:4566 s3://capstone/ID1/credit_card_churn_2022-08-07.csv

	aws kinesis create-stream --endpoint-url=http://localhost:4566 --stream-name predictions --shard-count 1 || true
	aws kinesis create-stream --endpoint-url=http://localhost:4566 --stream-name results --shard-count 1 || true

apply-model-train-flow:
	docker cp ./orchestration/model_train_flow.py prefect_agent_1:model_train_flow.py
	docker exec prefect_agent_1 prefect deployment create model_train_flow.py


# make setup-model-serve env=local
setup-model-serve:
	chmod 777 monitoring/
	chmod 777 monitoring/**/*
	cp .env.$(env) .env
	docker compose -f docker-compose-serve.yml  up --build --force-recreate

apply-prediction-reporting:
	docker cp ./monitoring/create_report.py prefect_agent_1:create_report.py
	docker exec prefect_agent_1 prefect deployment create create_report.py


stop-serve:	
	docker stop send_data-api
	docker stop send_data-kinesis
	docker stop serve
	docker rm -f serve

# make start-serve env=local
start-serve:
	cp .env.$(env) .env
	docker compose -f docker-compose-serve.yml up -d



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
