#Docker file for GSAreport tool name:emeraldit/gsareport
#docker run -v `pwd`/output:/output emeraldit/gsareport
#docker run -it -w /home/user -v `pwd`/src:/home/user tiagopeixoto/graph-tool bash
FROM tiagopeixoto/graph-tool:latest

COPY src /home/user
WORKDIR /home/user

#RUN pacman -S sudo
RUN pacman --noconfirm -S python-pip
RUN pacman --noconfirm -S gcc
RUN pacman --noconfirm -S nodejs npm
RUN pacman --noconfirm -S git
RUN pacman --noconfirm -S binutils

RUN pip install -r requirements.txt
RUN pip install pyinstaller

ENTRYPOINT ["python","./GSAreport.py"]
