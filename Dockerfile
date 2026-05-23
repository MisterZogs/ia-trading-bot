FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PAPER_TRADING=true
ENV EXCHANGE=binance

CMD ["python3", "main.py"]
