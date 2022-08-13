import os
import json
from time import sleep

import pyarrow.csv as pcsv
import requests

SERVE_ADDRESS = os.getenv('SERVE_ADDRESS', 'http://127.0.0.1:9696/predict')

table = pcsv.read_csv("../input_clean/credit_card_churn_clean.csv")
data = table.to_pylist()
# {'': 0, 'customer_id': 768805383, 'customer_age': 45, 'gender': 'M', 'dependent_count': 3, 'education_level': 1, 'marital_status': 'married', 
# 'income_category': 2, 'card_category': 'blue', 'months_on_book': 39, 'total_relationship_count': 5, 'credit_limit': 12691.0, 
# 'total_revolving_bal': 777, 'active_customer': True}

for row in data:

    payload = {
      'customer_age': row['customer_age'],
      'gender': row['gender'],
      'dependent_count': row['dependent_count'],
      'education_level': row['education_level'],
      'marital_status': row['marital_status'],
      'income_category': row['income_category'],
      'card_category': row['card_category'],
      'months_on_book': row['months_on_book'],
      'total_relationship_count': row['total_relationship_count'],
      'credit_limit': row['credit_limit'],
      'total_revolving_bal': row['total_revolving_bal'],
    }

    while True:
      print(f'Request: {payload}')
      resp = requests.post(SERVE_ADDRESS,
                              headers={"Content-Type": "application/json"},
                              data=json.dumps(row)) \
                      .json()
      print(f"Response: {resp}\n")
      sleep(30*1)
