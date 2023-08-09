FROM python:3.10-slim-buster

RUN pip install --no-cache-dir pandas numpy scikit-learn joblib explainerdashboard dill

COPY eda.py ./
COPY model.py ./
COPY dashboard.py ./
COPY app.py ./

RUN python eda.py
RUN python model.py
RUN python dashboard.py 

EXPOSE 5000
CMD ["python", "./app.py"]


