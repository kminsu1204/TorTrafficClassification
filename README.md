# Tor Traffic Classification from Raw Packet Header using Convolutional Neural Network
This repository provides the code used in the following paper:

M. Kim and A. Anpalagan, "Tor Traffic Classification from Raw Packet Header using Convolutional Neural Network," 
2018 1st IEEE International Conference on Knowledge Innovation and Invention (ICKII), Jeju, 2018, pp. 187-190, 
doi: 10.1109/ICKII.2018.8569113.

https://ieeexplore.ieee.org/document/8569113/

## Get Started
Tor/Non-Tor Dataset is available at [UNB-CIC](https://www.unb.ca/cic/).

## Data Parser
The initial dataset is given in PCAP format that can be efficiently read in Wireshark software.

Using files in **pcapFileConverters**, Tor / non-Tor traffic packets can be extracted into hexadecimal values in .txt file.

**RawPacketParser** parses raw data and extracts TCP/IP packet header values from it.

## Data Classifier
**RawPacketClassifier** is implemented using Tensorflow and the model runs on Google Cloud Platform.

*gcloud ml-engine execute command.txt* is the command used to run the model using ML-Engine.

*DatasetProcessor* processes data for testing various scenarios used in the paper. 
Please note that this research doesn't consider time-series feature to divide training/test set, so the result might be different if dataset is divided based on it including accuracy.


