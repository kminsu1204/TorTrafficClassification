from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import random
import pandas as pd
from google.cloud import storage
import io

class Dataset:
    def __init__(self, data_files_path, sampling_size, test_size):
        self.data_files_path = data_files_path # path to dataset (local or GCS)
        self.sampling_size = sampling_size  # how much data do you want to use? e.g) 0.1 means using 10% of dataset from total dataset 
        self.test_size = test_size  # how much do you want to use as testset from dataset? e.g) 0.1 is 10%
        self._get_data()

    def read_csv_file(self, file_name, sampling_size):
        bucket_name = 'crypto-airlock-199321-ml'
        directory_path = 'RawPacketClassifier/input/'
        
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = storage.Blob(directory_path+file_name, bucket)   # get blob from GCS bucket
        temp = blob.download_as_string().decode("utf-8")    # decoding from byte format
        
        n = temp.count('\r')    # count number of packetse
        samples = random.sample(range(n), n - sampling_size)

        return pd.read_csv(io.StringIO(temp), delimiter=',', dtype=np.float32, header=None, skiprows=samples)

    def _get_data(self):
        # 1.Read dataset files
        # Tor: 11,743,657  nonTor: 9,342,053  Total: 21,085,710 (4.76 GB, 75 files)
        print('Reading Tor csv files...')
        CHAT_facebookchatgateway = self.read_csv_file(r'CHAT_facebookchatgateway.csv', self.sampling_size)  # 5406
        CHAT_gate_AIM_chat = self.read_csv_file(r'CHAT_gate_AIM_chat.csv', self.sampling_size)  # 580
        print('tor chat1: '+repr(CHAT_facebookchatgateway.shape) + ',tor chat2: '+repr(CHAT_gate_AIM_chat.shape))

        print('Reading Chat related csv files...')
        AIM_Chat = self.read_csv_file(r'AIM_Chat.csv', self.sampling_size)   # 526
        aimchat = self.read_csv_file(r'aimchat.csv', self.sampling_size) # 190
        #print('nontor chat1: '+repr(AIM_Chat)+', nontor chat2: '+repr(aimchat))

        # 2.Split datasets into the training and testing sets (Shuffle is True by default)
        print('Processing Tor files... (train/test split + object concatenation)')
        CHAT_facebookchatgateway_train, CHAT_facebookchatgateway_test = train_test_split(CHAT_facebookchatgateway, test_size=self.test_size)
        CHAT_gate_AIM_chat_train, CHAT_gate_AIM_chat_test = train_test_split(CHAT_gate_AIM_chat, test_size=self.test_size)
        print('tor chat train: '+repr(CHAT_facebookchatgateway_train.shape))

        print('Processing nonTor files... (train/test split + object concatenation)')
        AIM_Chat_train, AIM_Chat_test = train_test_split(AIM_Chat, test_size=self.test_size)
        aimchat_train, aimchat_test = train_test_split(aimchat, test_size=self.test_size)
  
        # 3.Merge training and testing sets respectively.
        concatenated_tor_train_set = np.concatenate((CHAT_facebookchatgateway_train, CHAT_gate_AIM_chat_train), axis=0)
        
        concatenated_tor_test_set = np.concatenate((CHAT_facebookchatgateway_test, CHAT_gate_AIM_chat_test), axis=0)

        concatenated_nonTor_train_set = np.concatenate((AIM_Chat_train, aimchat_train), axis=0)

        concatenated_nonTor_test_set = np.concatenate((AIM_Chat_test, aimchat_test), axis=0)

        concatenated_train_set = np.concatenate((concatenated_tor_train_set, concatenated_nonTor_train_set), axis=0)
        concatenated_test_set = np.concatenate((concatenated_tor_test_set, concatenated_nonTor_test_set), axis=0)

        concatenated_train_set = shuffle(concatenated_train_set)
        concatenated_test_set = shuffle(concatenated_test_set)

        print('concatenated train set: '+repr(concatenated_train_set.shape))

        self.x_train = concatenated_train_set[:, 0:-1]
        self.y_train = concatenated_train_set[:, [-1]] # 0 ~ 15
        self.train_length = self.x_train.shape[0]
        print('train length: '+repr(self.train_length))
        self.x_test = concatenated_test_set[:, 0:-1]
        self.y_test = concatenated_test_set[:, [-1]]
        self.test_length = self.x_test.shape[0]

