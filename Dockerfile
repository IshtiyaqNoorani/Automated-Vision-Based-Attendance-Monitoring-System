FROM python:3.10

WORKDIR /app

# Install system dependencies required for insightface
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "web_app:app", "--bind", "0.0.0.0:5000"]