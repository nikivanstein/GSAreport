#Docker file for GSAreport tool name:emeraldit/gsareport
#docker run -v `pwd`/output:/output emeraldit/gsareport
#docker run -it -w /home/user -v `pwd`/src:/home/user tiagopeixoto/graph-tool bash
FROM tiagopeixoto/graph-tool:release-2.58

COPY src /home/user
WORKDIR /home/user

#RUN pacman -S sudo
RUN pacman --noconfirm -S python-pip
RUN pacman --noconfirm -S gcc
RUN pacman --noconfirm -S nodejs npm
RUN pacman --noconfirm -S git
RUN pacman --noconfirm -S binutils

#upgrade pip
#RUN python -m venv /home/user/venv
#RUN pip install --upgrade pip
RUN pip install -r requirements.txt --break-system-packages
RUN pip install pyinstaller --break-system-packages

ENTRYPOINT ["python","./GSAreport.py"]
