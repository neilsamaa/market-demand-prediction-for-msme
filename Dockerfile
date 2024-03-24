FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN python3 -m pip install statsmodels

RUN pip3 install --no-cache-dir streamlit pandas numpy
#RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "previous_sales.py", "--server.port=8501", "--server.address=0.0.0.0"]
