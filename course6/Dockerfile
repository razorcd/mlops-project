FROM public.ecr.aws/lambda/python:3.8

RUN pip install -U pip
RUN pip install pipenv 

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "lambda_function.py", "model.py", "lin_reg.bin", "./" ]

CMD [ "lambda_function.lambda_handler" ]
