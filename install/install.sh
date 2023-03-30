#sudo apt-get isntall make gcc

#sudo apt-get remove --purge python3-pip
sudo apt-get -y install python3-pip

pip3 install bitsandbytes
pip3 install -q datasets loralib sentencepiece
pip3 install -q git+https://github.com/zphang/transformers@c3dc391
pip3 install -q git+https://github.com/huggingface/peft.git