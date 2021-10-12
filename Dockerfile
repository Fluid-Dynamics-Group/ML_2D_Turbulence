FROM ubuntu:20.04

RUN apt-get update -y
RUN apt-get install software-properties-common -y
RUN add-apt-repository universe 
RUN add-apt-repository multiverse

RUN apt-get install libfftw3-dev build-essential gfortran python3-pip -y

RUN pip3 install numpy matplotlib

COPY . /turbulence

WORKDIR /turbulence

RUN sh ./Clean_Compile.sh

CMD python3 "./2D_Turbulence_ML.py"
