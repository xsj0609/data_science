# apt-get install python-pip python-dev build-essential
cp /etc/apt/sources.list /etc/apt/sources.list.bak
cat>/etc/apt/sources.list<<EOF
deb http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse
EOF
apt-get update

# pip install --upgrade pip
# pip install --upgrade virtualenv
mkdir ~/.pip
cat>~/.pip/pip.conf<<EOF
[global]
index-url = http://mirrors.aliyun.com/pypi/simple/ 
[install]
trusted-host=mirrors.aliyun.com
EOF