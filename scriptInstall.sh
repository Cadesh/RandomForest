#!/bin/bash
# author: Andre Rosa
# objective: Script to install VubbleVideoCategorizer Server WITHIN BITBUCKET PIPELINES
# THIS SCRIPT USES SELF-SIGNED SSL CERTIFICATES. 
# IF YOU CHANGED THE FILE IN WINDOWS, USE THE COMMAND BELLOW BEFORE RUNNING THE SCRIPT
# sed -i 's/\r//' script_name.sh 
# TO RUN THE SCRIPT USE A CLEAN D.O. SERVLET WITH UBUNTU 16.04.6 and SUDO POWERS
# sudo bash script_name.sh
#---------------------------------------------------------------------------

# GETS ARGUMENTS DOMAIN MUST BE IP, SECOND ARGUMENT IS CREATOR EMAIL
REPO='https://github.com/Cadesh/RandomForrest.git'  # git bitbucket repository origin of REAL code

echo -e "\e[96mINITIATE SCRIPT\e[39m"

#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# INSTALL ESSENTIALS - C++ MAKE
echo -e "\e[96mINSTALL UBUNTU ESSENTIALS (MAKE)\e[39m"
apt-get install build-essential -y
#---------------------------------------------------------------------------
#

#---------------------------------------------------------------------------
echo -e "\e[96mINSTALL GIT\e[39m"
sleep 2
apt-get install git
echo
#---------------------------------------------------------------------------

# INSTALL PYTHON3
echo -e "\e[96mINSTALL PYTHON3\e[39m"
apt-get update
apt-get -y upgrade
apt-get install -y python3-pip

apt-get install -y build-essential libssl-dev libffi-dev
apt-get install -y libsm6 libxrender1 libfontconfig1 libxext6 libxrender-dev

# DOWNLOAD CODE FROM BITBUCKET
echo -e "\e[96mCLONING BITBUCKET REPO\e[39m"
git clone $REPO
sleep 2
#---------------------------------------------------------------------------
#
#---------------------------------------------------------------------------
echo -e "\e[96mINSTALL PYTHON VIRTUAL ENV\e[39m"
#CREATE PYTHON ENVIRONMENT
sudo -H pip3 install --upgrade pip
sudo -H pip3 install virtualenv
cd pyCategorizer
echo -e "\e[96mCREATE PYTHON ENVIRONMENT\e[39m"
virtualenv catenv
source catenv/bin/activate
#INSTALL PYTHON MODULES
echo -e "\e[96mINSTALL PYTHON MODULES\e[39m"
pip install pandas
pip install numpy
pip install matplotlib
pip install scikit-learn
#---------------------------------------------------------------------------

# #---------------------------------------------------------------------------
# #---------------------------------------------------------------------------