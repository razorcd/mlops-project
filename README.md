# MLOPS training based on MLOPS Zoomcamp course offered by DataTalks.Club
https://github.com/DataTalksClub/mlops-zoomcamp

## Final project:


## Notes:
- MLOPS maturity model: https://docs.microsoft.com/en-us/azure/architecture/example-scenario/mlops/mlops-maturity-model


## Prefect setup
 - start prefect using Docker: 
    - `cd course3`
    - `in Dockerfile, edit 127.0.0.1 to server's public IP
    - `docker build -t prefect_test3 .`
    - ` docker run --rm --name prefectTest3 -p 4200:4200 -p 8080:8080 prefect_test3`
 - enable TCP and UDP for port 4200 in firewall

- config local machine: `prefect config set PREFECT_API_URL="http://<external-ip>:4200/api`"
 - `prefect config view` -> `PREFECT_API_URL='http://127.0.0.1:4200/api' (from profile)`
 - open `<external-ip>:4200` in browser

