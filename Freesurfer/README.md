# Ubuntu
1. Download freesurfer from <a href="https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall">here</a> (use `wget` terminal command).
2. Unzip: `sudo tar -C /usr/local -xzvf downloaded_file`
3. Install requirements on VM:
	```sudo apt-get update
	sudo apt-get install git
	sudo apt-get install python-pip
	sudo pip install --upgrade pip
	sudo pip install --upgrade setuptools
	sudo easy_install nipype
	sudo apt-get install csh
	sudo apt-get install tcsh
	```

4. SSH to host (Remember to add your machine in hosts)
     1. `sudo nano /etc/hosts`
     2. Add: `127.0.1.1 my-machine` in the file.
     
5. Change access permission to save preprocessed subjects data in subjects folder.
	```
	sudo chmod 777 /usr/local/freesurfer/subjects
	```

6. Get a license and cope it in Freesurfer install folder:
	```
	sudo cp license.txt /usr/local/freesurfer
	```

5. export and source:
	```
	export FREESURFER_HOME=/usr/local/freesurfer
	source $FREESURFER_HOME/SetUpFreeSurfer.sh
	```
