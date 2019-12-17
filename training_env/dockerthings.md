
### 1) Get Docker
From here:
https://www.docker.com/products/docker-desktop

### 2) Build the Docker image

##### Build and tag the Docker image from the instructions in Dockerfile.
Do this every time there is a change to the Dockerfile (like you need to add a Python library). Do this from the directory containing the Dockerfile (training_env by default).

You might need to sudo for this.

<pre><code>docker build -t cheesebot -f Dockerfile .
</pre></code>

This will take ~5 minutes on first run and end with <i>Successfully tagged cheesebot:latest</i>


### 3) Run the image
Replace ~/Documents/Git/Robotics/training_env/ with your local path. This example connects the training_env folder on your computer to the /tf/notebooks folder inside the Docker container.

<pre><code>docker run -it -v ~/Documents/Git/Robotics/training_env/:/tf/notebooks -p 8888:8888 cheesebot:latest
</pre></code>
You can now share files from your computer to the container through this folder connection.

If you don't want Docker to have access to anything on your computer, use this instead:

<pre><code>docker run -it -p 8888:8888 cheesebot:latest
</pre></code>


### 4) Open Jupyter Notebook
###### CLI outputs a URL like below.
Go there.
http://127.0.0.1:8888/?token=WHATEVERISHERE

You can run the Jupyter Notebook by clicking "Run All" from the Cell menu dropdown. You can also choose to run each section (cell) one at a time.
###### (Done)


## Other run commands
https://docs.docker.com/engine/reference/run/

###### Stop all docker containers (just in case)
<pre><code>docker kill $(docker ps -q)
</pre></code>

### AWS things not related to running
###### Retrieve login command to use to authenticate Docker client to registry (needs IAM credentials)
<pre><code>$(aws2 ecr get-login --no-include-email --region us-west-2)
</pre></code>



### I need different packages than the Docker build comes with
Modify Dockerfile and then re-build the container. All you need to do is type the commands you would use to install those packages from the command line. Then rebuild the Docker container.


### I'm a huge baby who doesn't want to use Docker
Inside the file 'Dockerfile' is a list of all the commands Docker uses to build the environment.


### This is way too much work and/or too complicated

Lucky for you, there's a notebook set up with everything ready to go:

http://notebook.kai.engineering/

There's a catch. Amazon charges me to run jobs on this notebook. You want to use it, you pay the costs. Email me and I'll generate a token for you.



http://github.com/kaibrooks
