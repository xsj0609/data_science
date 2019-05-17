# cd ~
# 让ssh保持连接
cat>>/etc/ssh/ssh_config<<EOF
ServerAliveInterval 60
ServerAliveCountMax 3
EOF

sh ~/sources.sh

apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
apt-key fingerprint 0EBFCD88
add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
apt-get update

apt-get install docker-ce docker-ce-cli containerd.io
# apt-cache madison docker-ce
# apt-get install docker-ce=5:18.09.1~3-0~ubuntu-xenial docker-ce-cli=5:18.09.1~3-0~ubuntu-xenial containerd.io
# docker run hello-world

# If you have nvidia-docker 1.0 installed: we need to remove it and all existing GPU containers
docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
apt-get purge -y nvidia-docker
#
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update

apt-get install -y nvidia-docker2
pkill -SIGHUP dockerd

mkdir dockerfile
cd ~/dockerfile

cat>Dockerfile<<EOF
FROM ufoym/deepo
COPY sources.sh .
RUN buildDeps='gcc cmake make wget git' \
    && sh sources.sh \
    && apt-get install -y $buildDeps \
    && pip install jupyter \
    && cd ~ \
    && git clone --recursive https://github.com/dmlc/xgboost \
    && cd ~/xgboost \
    && mkdir build \
    && cd ~/xgboost/build \
    && cmake .. -DUSE_CUDA=ON \
    && make -j \
    && cd ~/xgboost/python-package \
    && python setup.py install

# 设置系统环境
# ENV JAVA_HOME /usr/lib/jvm/java-7-oracle/ 
EXPOSE 8888
EOF

docker build -t xsj0609/xsj_ml:deepo_v1 .
nvidia-docker run -it -p 8889:8888 xsj0609/xsj_ml:deepo_v1 /bin/bash
