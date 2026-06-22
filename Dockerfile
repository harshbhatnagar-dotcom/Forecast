FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 10000

CMD sh -c "streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT"
