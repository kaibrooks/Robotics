# don't blame me
#           -Kai
# this is what Docker runs to build the container

ARG UBUNTU_VERSION=18.04
FROM ubuntu:${UBUNTU_VERSION} as base
RUN apt-get update && apt-get install -y curl

RUN apt-get install -y git python3-pip

RUN pip3 install --upgrade pip
RUN apt-get install -y protobuf-compiler python-pil python-lxml python-scipy

RUN pip3 install tensorflow && \
    pip3 install numpy pandas scipy sklearn matplotlib seaborn jupyter pyyaml h5py && \
    pip3 install keras --no-deps && \
    pip3 install imutils && \
    pip3 install Pillow && \
    pip3 install awscli && \
    apt-get update && \
    apt-get install -y libsm6 libxext6 libxrender-dev && \
    apt-get install unzip && \
    pip3 install opencv-python

RUN mkdir -p /tf/training-demo && chmod -R a+rwx /tf/
RUN mkdir -p /tf/training-demo/data && chmod -R a+rwx /tf/
RUN mkdir -p /tf/training-demo/trained_models && chmod -R a+rwx /tf/


RUN mkdir /.local && chmod a+rwx /.local
RUN apt-get install -y --no-install-recommends wget

WORKDIR /tf/training-demo
RUN wget https://raw.githubusercontent.com/kaibrooks/Robotics/demo-files/training_env/cheesebotTraining.ipynb && \
    wget https://raw.githubusercontent.com/kaibrooks/Robotics/demo-files/training_env/imageGenerator.ipynb

# copy pictures for training, remove this for a smaller build
WORKDIR /tf/training-demo/data
RUN wget https://github.com/kaibrooks/Robotics/raw/demo-files/training_env/data/imageset.zip
RUN unzip imageset.zip && rm imageset.zip

# copy a pre-trained model
WORKDIR /tf/training-demo/trained_models
RUN wget https://github.com/kaibrooks/Robotics/blob/demo-files/training_env/trained_models/2019-12-04_18-47-06_trainedmodel.h5 && \
    wget https://github.com/kaibrooks/Robotics/blob/demo-files/training_env/trained_models/2019-12-04_18-47-06_modelsummary.txt

RUN apt-get autoremove -y && apt-get remove -y wget
WORKDIR /tf

EXPOSE 8888

RUN python3 -m ipykernel.kernelspec

#RUN ["apt-get", "install", "-y", "libsm6", "libxext6", "libxrender-dev"]

CMD ["jupyter", "notebook", "--allow-root", "--notebook-dir=/tf", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
