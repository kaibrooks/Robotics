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
    pip3 install awscli

RUN mkdir -p /tf/tensorflow-tutorials && chmod -R a+rwx /tf/
RUN mkdir /.local && chmod a+rwx /.local
RUN apt-get install -y --no-install-recommends wget
WORKDIR /tf/tensorflow-tutorials
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/classification.ipynb
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/overfit_and_underfit.ipynb
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/regression.ipynb
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/save_and_load.ipynb
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/text_classification.ipynb
RUN wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/keras/text_classification_with_hub.ipynb

RUN apt-get autoremove -y && apt-get remove -y wget
WORKDIR /tf

EXPOSE 8888

RUN python3 -m ipykernel.kernelspec

CMD ["jupyter", "notebook", "--allow-root", "--notebook-dir=/tf", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
