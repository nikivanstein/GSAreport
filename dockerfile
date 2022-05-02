#Docker file for GSAreport tool name:emeraldit/gsareport
#docker run -v `pwd`/output:/output emeraldit/gsareport
#docker run -it -w /home/user -v `pwd`/src:/home/user tiagopeixoto/graph-tool bash
FROM tiagopeixoto/graph-tool:latest

COPY src /home/user
WORKDIR /home/user

#RUN pacman -S sudo
RUN pacman --noconfirm -S python-pip
RUN pacman --noconfirm -S nodejs npm
RUN cd savvy \
    && python setup.py install \
    && cd ..

RUN pip install -r requirements.txt

ENTRYPOINT ["python","./GSAreport.py"]
