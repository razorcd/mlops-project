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
	@echo "\nCommands:\n"
	@cat Makefile | egrep -e '^#run:.*'| sed -e 's~#~~g'
# @make -qpRr | egrep -e '^[a-z].*:' | sed -e 's~:~~g' | sort
	@echo ""

#run: make run_tests   to run tests locally
run_tests:
	pipenv run pytest server/tests	

#run: make reset_all   to delete all countainers and cleanup volumes
reset_all:
	docker compose -f docker-compose-serve.yml down
	docker compose -f docker-compose-model-registry.yml down
	rm -rf /tmp/mlopsdb
	rm -rf /tmp/mlopsartifacts  
	rm -rf /tmp/store   
	rm -rf /tmp/serve   
	rm -rf /tmp/mlreports
	rm .env

#run: make setup-model-registry env=local (or env=cloud)   to start model registry and training containers
setup-model-registry:
	mkdir -p /tmp/mlopsdb
	mkdir -p /tmp/mlopsartifacts  
	mkdir -p /tmp/store   
	mkdir -p /tmp/serve   
	mkdir -p /tmp/mlreports
	cp .env.$(env) .env
	docker compose -f docker-compose-model-registry.yml up --build --force-recreate
	
#run: make init_aws  to setup and initialize AWS services (uses localstack container)
init_aws:
	aws s3 mb s3://capstone --endpoint-url=http://localhost:4566 || true
	aws s3 cp input_clean/credit_card_churn_clean.csv --endpoint-url=http://localhost:4566 s3://capstone/ID1/credit_card_churn_2022-08-07.csv

	aws kinesis create-stream --endpoint-url=http://localhost:4566 --stream-name predictions --shard-count 1 || true
	aws kinesis create-stream --endpoint-url=http://localhost:4566 --stream-name results --shard-count 1 || true

#run: make apply-model-train-flow   to apply the automated model training DAG 
apply-model-train-flow:
	docker cp ./orchestration/model_train_flow.py prefect_agent_1:model_train_flow.py
	docker exec prefect_agent_1 prefect deployment create model_train_flow.py


#run: make setup-model-serve env=local (or env=cloud)   to start the model serving containers
setup-model-serve:
	chmod 777 monitoring/
	chmod 777 monitoring/**/*
	cp .env.$(env) .env
	docker compose -f docker-compose-serve.yml  up --build --force-recreate

#run: make apply-prediction-reporting   to apply the automated prediction reporting DAG
apply-prediction-reporting:
	docker cp ./monitoring/create_report.py prefect_agent_1:create_report.py
	docker exec prefect_agent_1 prefect deployment create create_report.py

#run: make stop-serve   to stop the model servers (http api and Stream)
stop-serve:	
	docker stop send_data-api
	docker stop send_data-kinesis
	docker stop serve
	docker rm -f serve

# make start-serve env=local   to start the model servers (http api and Stream)
start-serve:
	cp .env.$(env) .env
	docker compose -f docker-compose-serve.yml up -d
