
# 1) Get Docker
From here:
https://www.docker.com/products/docker-desktop

---
# 2) Build the Docker image

###### Build and tag the Docker image from the instructions in Dockerfile.
Do this every time there is a change to the Dockerfile (like you need to add a Python library). Do this from the directory containing the Dockerfile (training_env by default).

You might need to sudo for this.

<pre><code>docker build -t cheesebot -f Dockerfile .
</pre></code>

This will take ~5 minutes on first run and end with <b>Successfully tagged cheesebot:latest</b>

---
# 3) Run the image

Different settings to run the container below

###### Without local access and resets everything to default on exit:
<i>Run this if you don't know what you're doing since you can't screw up the files in the image</i>
<pre><code>docker run -it --rm -p 8888:8888 cheesebot:latest
</pre></code>

###### With persistent storage, without local folder access:
<pre><code>docker run -it -p 8888:8888 cheesebot:latest
</pre></code>

######  With persistant storage, with local folder access:
Replace ~/Documents/Git/Robotics/training_env/ with your local path
<pre><code>docker run -it -v ~/Documents/Git/Robotics/training_env/:/tf/notebooks -p 8888:8888 cheesebot:latest
</pre></code>
---
# 4) Open Jupyter Notebook
###### CLI outputs a URL like below.
Go there.
http://127.0.0.1:8888/?token=SOMEBULLSHITHERE

You can run the Jupyter Notebook by clicking "Run All" from the Cell menu dropdown.
###### (Done)

---
# Other run commands
https://docs.docker.com/engine/reference/run/

###### Stop all docker containers (just in case)
<pre><code>docker kill $(docker ps -q)
</pre></code>
---
# AWS things not related to running
###### Retrieve login command to use to authenticate Docker client to registry (needs IAM credentials)
<pre><code>$(aws2 ecr get-login --no-include-email --region us-west-2)
</pre></code>

---

# I'm a huge baby who doesn't want to use Docker
Here's the packages Docker builds with. This probably won't work right. I haven't tested this and have no sympathy if it fails:
<pre><code>apt-get update && apt-get install -y curl
apt-get install -y git python3-pip
pip3 install --upgrade pip
apt-get install -y protobuf-compiler python-pil python-lxml python-scipy
pip3 install tensorflow
pip3 install numpy pandas scipy sklearn matplotlib seaborn jupyter pyyaml h5py
pip3 install keras --no-deps
pip3 install imutils
pip3 install Pillow
pip3 install awscli
apt-get install unzip
</code></pre>

---

http://github.com/kaibrooks
