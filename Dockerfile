FROM python

COPY ./req.txt .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN python -m pip install --upgrade pip
RUN pip install -r req.txt
RUN pip install opencv-python-headless

ADD ./car_detection.py .
ADD dashboard.py .

CMD [ "python", "-u", "dashboard.py"]