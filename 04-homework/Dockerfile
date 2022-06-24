FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

RUN apt-get upgrade
RUN pip install -U pip
RUN pip install pipenv
RUN pip install pandas
RUN pip install pyarrow

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY ["predict.py", "model.bin", "./" ]

RUN python predict.py "2021" "04"


