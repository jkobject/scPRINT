```bash
sudo apt-get --purge remove '*nvidia*'
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install linux-headers-$(uname -r)
sudo apt-get remove --purge 'nvidia-*'
sudo apt-get autoremove
sudo apt-get clean
apt search nvidia-driver
# find the right one
sudo apt-get install nvidia-driver-555-open
sudo reboot
# reconnect
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
# find the right version
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get install cuda-11-8
sudo add-apt-repository ppa:flexiondotorg/nvtop
sudo apt install nvtop
```