### Start server:

```
gunicorn --bind=0.0.0.0:9696 --chdir=server --log-level=debug  predict:app
```

### Predict API call

```
curl -X POST -H 'Content-Type: application/json' localhost:9696/predict -d '{"customer_age":100,"gender":"F","dependent_count":2,"education_level":2,"marital_status":"married","income_category":2,"card_category":"blue","months_on_book":6,"total_relationship_count":3,"credit_limit":4000,"total_revolving_bal":2500}'
```