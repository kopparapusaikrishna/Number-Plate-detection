FROM python:3.10.6

COPY . /myapp
WORKDIR /myapp

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1-mesa-glx

RUN pip3 install -r requirements.txt

EXPOSE 5000
CMD ["python3", "app.py"]