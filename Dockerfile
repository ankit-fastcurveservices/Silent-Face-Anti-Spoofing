FROM python:3.9

RUN mkdir -p /home/liveness-svc/
COPY .  /home/liveness-svc/
WORKDIR /home/liveness-svc/

RUN apt-get update -y
RUN apt install libgl1-mesa-glx -y
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y
RUN pip3 install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install flask

CMD python3 liveness_classifier.py