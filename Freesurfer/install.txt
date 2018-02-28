#1- Download freesurfer: https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall
	wget "ftp://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/6.0.0/freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.0.tar.gz"

#2- Unzip
	sudo tar -C /usr/local -xzvf freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.0.tar.gz



4- Install requirements on VM:
	sudo apt-get update
	sudo apt-get install git
	sudo apt-get install python-pip
	sudo pip install --upgrade pip
	sudo pip install --upgrade setuptools
	sudo easy_install nipype

	source $FREESURFER_HOME/SetUpFreeSurfer.sh
	sudo apt-get install csh
	sudo apt-get install tcsh

5- 
	SSH to host:
	sudo nano /etc/hosts
	Add:
		127.0.1.1 my-machine
	sudo chmod 777 /usr/local/freesurfer/subjects 

6- copy the licenese using:
	sudo cp license.txt /usr/local/freesurfer

5- export and source
	export FREESURFER_HOME=/usr/local/freesurfer
	source $FREESURFER_HOME/SetUpFreeSurfer.sh