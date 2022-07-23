Create s3 bucket in localstack:
```
aws s3 mb s3://nyc-duration --endpoint-url=http://localhost:4566
```

List files in s3 bucket:
```
aws s3 ls --endpoint-url=http://localhost:4566 s3://nyc-duration/ID1 --recursive --human-readable --summarize
```