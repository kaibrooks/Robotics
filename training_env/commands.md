# docker things
###### build and tag the docker image from Dockerfile
###### do this every time there is a change to the Dockerfile
###### you might need to sudo
docker build -t cheesebot -f Dockerfile .

###### run the above image we just built
###### replace ~/Documents/Git/Robotics/training_env/ with your local path
###### if you don't want it to have local access, remove ~localpath:/tf/notebooks
###### no local access / with local access below
docker run -it --rm -v -p 8888:8888 cheesebot:latest
docker run -it --rm -v ~/Documents/Git/Robotics/training_env/:/tf/notebooks -p 8888:8888 cheesebot:latest

###### if all goes well, CLI outputs a URL like below. go there
http://127.0.0.1:8888/?token=SOMEBULLSHITHERE

###### stop all docker containers just in case
docker kill $(docker ps -q)

# AWS things not related to cheesebot running
###### retrieve login command to use to authenticate Docker client to registry (needs iam creds)
$(aws2 ecr get-login --no-include-email --region us-west-2)
