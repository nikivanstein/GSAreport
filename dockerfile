#Docker file for GSAreport tool
FROM python:3.8.18-slim

COPY src /home/user
WORKDIR /home/user

RUN apt-get update
RUN apt-get install -y python3-graph-tool
RUN apt-get install -y libcairo2-dev pkg-config python3-dev
RUN apt-get install -y nodejs

RUN pip install --upgrade pip
RUN pip install -r requirements.txt --break-system-packages
RUN pip install pyinstaller --break-system-packages

ENTRYPOINT ["python","./GSAreport.py"]
