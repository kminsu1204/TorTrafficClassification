from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import random
import pandas as pd
from google.cloud import storage
import io
#import tensorflow as tf

class Dataset:
    def __init__(self, data_files_path, sampling_size, test_size):
        self.data_files_path = data_files_path # path to dataset (local or GCS)
        self.sampling_size = sampling_size  # how much data do you want to use? e.g) 0.1 means using 10% of dataset from total dataset 
        self.test_size = test_size  # how much do you want to use as testset from dataset? e.g) 0.1 is 10%
        self._get_data()

    def read_csv_file(self, file_name):
        bucket_name = 'crypto-airlock-199321-ml'
        directory_path = 'RawPacketClassifier/input/'
        
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = storage.Blob(directory_path+file_name, bucket)   # get blob from GCS bucket
        temp = blob.download_as_string().decode("utf-8")    # decoding from byte format
        
        #n = temp.count('\r')    # count number of packetse
        #samples = random.sample(range(n), n - sampling_size)

        return pd.read_csv(io.StringIO(temp), delimiter=',', dtype=np.float32, header=None, skiprows=None)

    def _get_data(self):
        # 1.Read dataset files
        # Tor: 11,743,657  nonTor: 9,342,053  Total: 21,085,710 (4.76 GB, 75 files)
        print('Reading Tor csv files...')
        print('Reading Audio related csv files...')
        tor_Audio = np.concatenate((self.read_csv_file(r'AUDIO_spotifygateway.csv'), self.read_csv_file(r'AUDIO_tor_spotify.csv'), self.read_csv_file(r'AUDIO_tor_spotify2.csv')), axis=0)
        tor_Audio = shuffle(tor_Audio)
        print(tor_Audio.shape)
        
        print('Reading Browsing related csv files...')
        tor_Browsing = np.concatenate((self.read_csv_file(r'BROWSING_gate_SSL_Browsing.csv'), self.read_csv_file(r'BROWSING_ssl_browsing_gateway.csv'), self.read_csv_file(r'BROWSING_tor_browsing_ara.csv'), self.read_csv_file(r'BROWSING_tor_browsing_ger.csv'), self.read_csv_file(r'BROWSING_tor_browsing_mam.csv'), self.read_csv_file(r'BROWSING_tor_browsing_mam2.csv')), axis=0)
        tor_Browsing = shuffle(tor_Browsing)
        print(tor_Browsing.shape)        
        
        print('Reading Chat related csv files...')
        tor_Chat = np.concatenate((self.read_csv_file(r'CHAT_aimchatgateway.csv'), self.read_csv_file(r'CHAT_facebookchatgateway.csv'), self.read_csv_file(r'CHAT_gate_AIM_chat.csv'), self.read_csv_file(r'CHAT_gate_facebook_chat.csv'), self.read_csv_file(r'CHAT_gate_hangout_chat.csv'), self.read_csv_file(r'CHAT_gate_ICQ_chat.csv'), self.read_csv_file(r'CHAT_gate_skype_chat.csv'), self.read_csv_file(r'CHAT_hangoutschatgateway.csv'), self.read_csv_file(r'CHAT_icqchatgateway.csv'), self.read_csv_file(r'CHAT_skypechatgateway.csv')), axis=0)
        tor_Chat = shuffle(tor_Chat)
        print(tor_Chat.shape)

        print('Reading File transfer related csv files...')
        tor_File = np.concatenate((self.read_csv_file(r'FILE-TRANSFER_gate_FTP_transfer.csv'), self.read_csv_file(r'FILE-TRANSFER_gate_SFTP_filetransfer.csv'), self.read_csv_file(r'FILE-TRANSFER_tor_skype_transfer.csv')), axis=0)
        tor_File = shuffle(tor_File)
        print(tor_File.shape)

        print('Reading Mail related csv files...')
        tor_Email = np.concatenate((self.read_csv_file(r'MAIL_gate_Email_IMAP_filetransfer.csv'), self.read_csv_file(r'MAIL_gate_POP_filetransfer.csv'), self.read_csv_file(r'MAIL_Gateway_Thunderbird_Imap.csv'), self.read_csv_file(r'MAIL_Gateway_Thunderbird_POP.csv')), axis=0)
        tor_Email = shuffle(tor_Email)
        print(tor_Email.shape)

        print('Reading P2P related csv files...')
        tor_P2P = np.concatenate((self.read_csv_file(r'P2P_tor_p2p_multipleSpeed.csv'), self.read_csv_file(r'P2P_tor_p2p_vuze.csv')), axis=0)
        tor_P2P = shuffle(tor_P2P)
        print(tor_P2P.shape)

        print('Reading Video related csv files...')
        tor_Video = np.concatenate((self.read_csv_file(r'VIDEO_Vimeo_Gateway.csv'), self.read_csv_file(r'VIDEO_Youtube_Flash_Gateway.csv'), self.read_csv_file(r'VIDEO_Youtube_HTML5_Gateway.csv')), axis=0)
        tor_Video = shuffle(tor_Video)
        print(tor_Video.shape)

        print('Reading VoIP related csv files...')
        tor_VoIP = np.concatenate((self.read_csv_file(r'VOIP_Facebook_Voice_Gateway.csv'), self.read_csv_file(r'VOIP_gate_facebook_Audio.csv'), self.read_csv_file(r'VOIP_gate_hangout_audio.csv'), self.read_csv_file(r'VOIP_gate_Skype_Audio.csv'), self.read_csv_file(r'VOIP_Hangouts_voice_Gateway.csv'), self.read_csv_file(r'VOIP_Skype_Voice_Gateway.csv')), axis=0)
        tor_VoIP = shuffle(tor_VoIP)
        print(tor_VoIP.shape)
        print(tor_Audio.shape[0] + tor_Browsing.shape[0] + tor_Chat.shape[0] + tor_File.shape[0] + tor_Email.shape[0] + tor_P2P.shape[0] + tor_Video.shape[0] + tor_VoIP.shape[0])

        
        print('Reading nonTor csv files...')
        print('Reading Audio related csv files...')
        nonTor_Audio = np.concatenate((self.read_csv_file(r'facebook_Audio.csv'), self.read_csv_file(r'Hangout_Audio.csv'), self.read_csv_file(r'Skype_Audio.csv'), self.read_csv_file(r'spotify.csv'), self.read_csv_file(r'spotify2.csv'), self.read_csv_file(r'spotifyAndrew.csv')), axis=0)
        nonTor_Audio = shuffle(nonTor_Audio)
        print(nonTor_Audio.shape)

        print('Reading Browsing related csv files...')
        nonTor_Browsing = np.concatenate((self.read_csv_file(r'browsing.csv'), self.read_csv_file(r'browsing_ara.csv'), self.read_csv_file(r'browsing_ara2.csv'), self.read_csv_file(r'browsing_ger.csv'), self.read_csv_file(r'browsing2.csv'), self.read_csv_file(r'ssl.csv'), self.read_csv_file(r'SSL_Browsing.csv')), axis=0)
        nonTor_Browsing = shuffle(nonTor_Browsing)
        print(nonTor_Browsing.shape)

        print('Reading Chat related csv files...')
        nonTor_Chat = np.concatenate((self.read_csv_file(r'AIM_Chat.csv'), self.read_csv_file(r'aimchat.csv'), self.read_csv_file(r'facebook_chat.csv'), self.read_csv_file(r'facebookchat.csv'), self.read_csv_file(r'hangout_chat.csv'), self.read_csv_file(r'hangoutschat.csv'), self.read_csv_file(r'ICQ_Chat.csv'), self.read_csv_file(r'icqchat.csv'), self.read_csv_file(r'skype_chat.csv'), self.read_csv_file(r'skypechat.csv')), axis=0)
        nonTor_Chat = shuffle(nonTor_Chat)
        print(nonTor_Chat.shape)

        print('Reading File transfer related csv files...')
        nonTor_File = np.concatenate((self.read_csv_file(r'FTP_filetransfer.csv'), self.read_csv_file(r'SFTP_filetransfer.csv'), self.read_csv_file(r'skype_transfer.csv')), axis=0)
        nonTor_File = shuffle(nonTor_File)
        print(nonTor_File.shape)

        print('Reading Mail related csv files...')
        nonTor_Email = np.concatenate((self.read_csv_file(r'Email_IMAP_filetransfer.csv'), self.read_csv_file(r'POP_filetransfer.csv'), self.read_csv_file(r'Workstation_Thunderbird_Imap.csv'), self.read_csv_file(r'Workstation_Thunderbird_POP.csv')), axis=0)
        nonTor_Email = shuffle(nonTor_Email)
        print(nonTor_Email.shape)

        print('Reading P2P related csv files...')
        nonTor_P2P = np.concatenate((self.read_csv_file(r'p2p_multipleSpeed.csv'), self.read_csv_file(r'p2p_vuze.csv') ), axis=0)
        nonTor_P2P = shuffle(nonTor_P2P)
        print(nonTor_P2P.shape)

        print('Reading Video related csv files...')
        nonTor_Video = np.concatenate((self.read_csv_file(r'Vimeo_Workstation.csv'), self.read_csv_file(r'Youtube_Flash_Workstation.csv'), self.read_csv_file(r'Youtube_HTML5_Workstation.csv')), axis=0)
        nonTor_Video = shuffle(nonTor_Video)
        print(nonTor_Video.shape)

        print('Reading VoIP related csv files...')
        nonTor_VoIP = np.concatenate((self.read_csv_file(r'Facebook_Voice_Workstation.csv'), self.read_csv_file(r'Hangouts_voice_Workstation.csv'), self.read_csv_file(r'Skype_Voice_Workstation.csv')), axis=0)
        nonTor_VoIP = shuffle(nonTor_VoIP)
        print(nonTor_VoIP.shape)
        print(nonTor_Audio.shape[0] + nonTor_Browsing.shape[0] + nonTor_Chat.shape[0] + nonTor_File.shape[0] + nonTor_Email.shape[0] + nonTor_P2P.shape[0] + nonTor_Video.shape[0] + nonTor_VoIP.shape[0])


        # 2.Split datasets into the training and testing sets (Shuffle is True by default)
        print('Processing Tor files... (train/test split + object concatenation)')
       
        '''tor_Audio[:, [-1]] = 1
        tor_Browsing[:, [-1]] = 1
        tor_Chat[:, [-1]] = 1
        tor_File[:, [-1]] = 1
        tor_Email[:, [-1]] = 1
        tor_P2P[:, [-1]] = 1
        tor_Video[:, [-1]] = 1
        tor_VoIP[:, [-1]] = 1'''

        tor_Audio_train, tor_Audio_test = train_test_split(tor_Audio[0:20000, :], test_size=self.test_size)
        tor_Browsing_train, tor_Browsing_test = train_test_split(tor_Browsing[0:20000, :], test_size=self.test_size)
        tor_Chat_train, tor_Chat_test = train_test_split(tor_Chat[0:20000, :], test_size=self.test_size)
        tor_File_train, tor_File_test = train_test_split(tor_File[0:20000, :], test_size=self.test_size)
        tor_Email_train, tor_Email_test = train_test_split(tor_Email[0:20000, :], test_size=self.test_size)
        tor_P2P_train, tor_P2P_test = train_test_split(tor_P2P[0:20000, :], test_size=self.test_size)
        tor_Video_train, tor_Video_test = train_test_split(tor_Video[0:20000, :], test_size=self.test_size)
        tor_VoIP_train, tor_VoIP_test = train_test_split(tor_VoIP[0:20000, :], test_size=self.test_size)

        print('Processing nonTor files... (train/test split + object concatenation)')

        '''nonTor_Audio[:, [-1]] = 2
        nonTor_Browsing[:, [-1]] = 2
        nonTor_Chat[:, [-1]] = 2
        nonTor_File[:, [-1]] = 2
        nonTor_Email[:, [-1]] = 2
        nonTor_P2P[:, [-1]] = 2
        nonTor_Video[:, [-1]] = 2
        nonTor_VoIP[:, [-1]] = 2'''

        nonTor_Audio_train, nonTor_Audio_test = train_test_split(nonTor_Audio[0:20000, :], test_size=self.test_size)
        nonTor_Browsing_train, nonTor_Browsing_test = train_test_split(nonTor_Browsing[0:20000, :], test_size=self.test_size)
        nonTor_Chat_train, nonTor_Chat_test = train_test_split(nonTor_Chat[0:20000, :], test_size=self.test_size)
        nonTor_File_train, nonTor_File_test = train_test_split(nonTor_File[0:20000, :], test_size=self.test_size)
        nonTor_Email_train, nonTor_Email_test = train_test_split(nonTor_Email[0:20000, :], test_size=self.test_size)
        nonTor_P2P_train, nonTor_P2P_test = train_test_split(nonTor_P2P[0:20000, :], test_size=self.test_size)
        nonTor_Video_train, nonTor_Video_test = train_test_split(nonTor_Video[0:20000, :], test_size=self.test_size)
        nonTor_VoIP_train, nonTor_VoIP_test = train_test_split(nonTor_VoIP[0:20000, :], test_size=self.test_size)

        '''nonTor_Audio_train, nonTor_Audio_test = train_test_split(nonTor_Audio[0:721, :], test_size=self.test_size)
        nonTor_Browsing_train, nonTor_Browsing_test = train_test_split(nonTor_Browsing[0:1604, :], test_size=self.test_size)
        nonTor_Chat_train, nonTor_Chat_test = train_test_split(nonTor_Chat[0:323, :], test_size=self.test_size)
        nonTor_File_train, nonTor_File_test = train_test_split(nonTor_File[0:864, :], test_size=self.test_size)
        nonTor_Email_train, nonTor_Email_test = train_test_split(nonTor_Email[0:282, :], test_size=self.test_size)
        nonTor_P2P_train, nonTor_P2P_test = train_test_split(nonTor_P2P[0:1085, :], test_size=self.test_size)
        nonTor_Video_train, nonTor_Video_test = train_test_split(nonTor_Video[0:874, :], test_size=self.test_size)
        nonTor_VoIP_train, nonTor_VoIP_test = train_test_split(nonTor_VoIP[0:2291, :], test_size=self.test_size)'''

        # 3.Merge training and testing sets respectively.
        print('Merge training and testing sets respectively...')
        concatenated_train_set = np.concatenate((nonTor_Audio_train, nonTor_Browsing_train, nonTor_Chat_train, nonTor_File_train, nonTor_Email_train, nonTor_P2P_train, nonTor_Video_train, nonTor_VoIP_train, tor_Audio_train, tor_Browsing_train, tor_Chat_train, tor_File_train, tor_Email_train, tor_P2P_train, tor_Video_train, tor_VoIP_train), axis=0)
        concatenated_test_set = np.concatenate((nonTor_Audio_test, nonTor_Browsing_test, nonTor_Chat_test, nonTor_File_test, nonTor_Email_test, nonTor_P2P_test, nonTor_Video_test, nonTor_VoIP_test, tor_Audio_test, tor_Browsing_test, tor_Chat_test, tor_File_test, tor_Email_test, tor_P2P_test, tor_Video_test, tor_VoIP_test), axis=0)
        
        '''concatenated_train_set = np.concatenate((nonTor_Audio_train, nonTor_Browsing_train, nonTor_Chat_train, nonTor_File_train, nonTor_Email_train, nonTor_P2P_train, nonTor_Video_train, nonTor_VoIP_train), axis=0)
        concatenated_test_set = np.concatenate((nonTor_Audio_test, nonTor_Browsing_test, nonTor_Chat_test, nonTor_File_test, nonTor_Email_test, nonTor_P2P_test, nonTor_Video_test, nonTor_VoIP_test), axis=0)'''

        concatenated_train_set = shuffle(concatenated_train_set)
        concatenated_test_set = shuffle(concatenated_test_set)

        self.x_train = concatenated_train_set[:, 0:-1]
        self.y_train = concatenated_train_set[:, [-1]] # 0 ~ 15
        self.train_length = self.x_train.shape[0]
        print('Number of data for train: '+repr(self.train_length))

        self.x_test = concatenated_test_set[:, 0:-1]
        self.y_test = concatenated_test_set[:, [-1]]
        self.test_length = self.x_test.shape[0]
        print('Number of data for test: '+repr(self.test_length))

        self.nonTor_Audio_x_test = nonTor_Audio_test[:, 0:-1]
        self.nonTor_Audio_y_test = nonTor_Audio_test[:, [-1]]

        self.nonTor_Browsing_x_test = nonTor_Browsing_test[:, 0:-1]
        self.nonTor_Browsing_y_test = nonTor_Browsing_test[:, [-1]]

        self.nonTor_Chat_x_test = nonTor_Chat_test[:, 0:-1]
        self.nonTor_Chat_y_test = nonTor_Chat_test[:, [-1]]

        self.nonTor_File_x_test = nonTor_File_test[:, 0:-1]
        self.nonTor_File_y_test = nonTor_File_test[:, [-1]]

        self.nonTor_Email_x_test = nonTor_Email_test[:, 0:-1]
        self.nonTor_Email_y_test = nonTor_Email_test[:, [-1]]

        self.nonTor_P2P_x_test = nonTor_P2P_test[:, 0:-1]
        self.nonTor_P2P_y_test = nonTor_P2P_test[:, [-1]]

        self.nonTor_Video_x_test = nonTor_Video_test[:, 0:-1]
        self.nonTor_Video_y_test = nonTor_Video_test[:, [-1]]

        self.nonTor_VoIP_x_test = nonTor_VoIP_test[:, 0:-1]
        self.nonTor_VoIP_y_test = nonTor_VoIP_test[:, [-1]]

        self.tor_Audio_x_test = tor_Audio_test[:, 0:-1]
        self.tor_Audio_y_test = tor_Audio_test[:, [-1]]

        self.tor_Browsing_x_test = tor_Browsing_test[:, 0:-1]
        self.tor_Browsing_y_test = tor_Browsing_test[:, [-1]]

        self.tor_Chat_x_test = tor_Chat_test[:, 0:-1]
        self.tor_Chat_y_test = tor_Chat_test[:, [-1]]

        self.tor_File_x_test = tor_File_test[:, 0:-1]
        self.tor_File_y_test = tor_File_test[:, [-1]]

        self.tor_Email_x_test = tor_Email_test[:, 0:-1]
        self.tor_Email_y_test = tor_Email_test[:, [-1]]

        self.tor_P2P_x_test = tor_P2P_test[:, 0:-1]
        self.tor_P2P_y_test = tor_P2P_test[:, [-1]]

        self.tor_Video_x_test = tor_Video_test[:, 0:-1]
        self.tor_Video_y_test = tor_Video_test[:, [-1]]

        self.tor_VoIP_x_test = tor_VoIP_test[:, 0:-1]
        self.tor_VoIP_y_test = tor_VoIP_test[:, [-1]]

        print('Organizing dataset done...')

