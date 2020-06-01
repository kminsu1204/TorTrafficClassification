from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import random
import pandas as pd
from google.cloud import storage
import io
import tensorflow as tf

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
        print('Reading Audio related csv files...')
        AUDIO_spotifygateway = self.read_csv_file(r'AUDIO_spotifygateway.csv', self.sampling_size)  # 35485 (number of packets)
        AUDIO_tor_spotify = self.read_csv_file(r'AUDIO_tor_spotify.csv', self.sampling_size)    # 35493
        AUDIO_tor_spotify2 = self.read_csv_file(r'AUDIO_tor_spotify2.csv', self.sampling_size)  # 159018
        
        print('Reading Browsing related csv files...')
        BROWSING_gate_SSL_Browsing = self.read_csv_file(r'BROWSING_gate_SSL_Browsing.csv', self.sampling_size)  # 117531
        BROWSING_ssl_browsing_gateway = self.read_csv_file(r'BROWSING_ssl_browsing_gateway.csv', self.sampling_size)    # 41235
        BROWSING_tor_browsing_ara = self.read_csv_file(r'BROWSING_tor_browsing_ara.csv', self.sampling_size)    # 99878
        BROWSING_tor_browsing_ger = self.read_csv_file(r'BROWSING_tor_browsing_ger.csv', self.sampling_size)    # 154079
        BROWSING_tor_browsing_mam = self.read_csv_file(r'BROWSING_tor_browsing_mam.csv', self.sampling_size)    # 170612
        BROWSING_tor_browsing_mam2 = self.read_csv_file(r'BROWSING_tor_browsing_mam2.csv', self.sampling_size)  # 43062
        
        print('Reading Chat related csv files...')
        CHAT_aimchatgateway = self.read_csv_file(r'CHAT_aimchatgateway.csv', self.sampling_size)    # 227
        CHAT_facebookchatgateway = self.read_csv_file(r'CHAT_facebookchatgateway.csv', self.sampling_size)  # 5406
        CHAT_gate_AIM_chat = self.read_csv_file(r'CHAT_gate_AIM_chat.csv', self.sampling_size)  # 580
        CHAT_gate_facebook_chat = self.read_csv_file(r'CHAT_gate_facebook_chat.csv', self.sampling_size)    # 2902
        CHAT_gate_hangout_chat = self.read_csv_file(r'CHAT_gate_hangout_chat.csv', self.sampling_size)  # 3566
        CHAT_gate_ICQ_chat = self.read_csv_file(r'CHAT_gate_ICQ_chat.csv', self.sampling_size)  # 417
        CHAT_gate_skype_chat = self.read_csv_file(r'CHAT_gate_skype_chat.csv', self.sampling_size)  # 1907
        CHAT_hangoutschatgateway = self.read_csv_file(r'CHAT_hangoutschatgateway.csv', self.sampling_size)  # 2948
        CHAT_icqchatgateway = self.read_csv_file(r'CHAT_icqchatgateway.csv', self.sampling_size)    # 534
        CHAT_skypechatgateway = self.read_csv_file(r'CHAT_skypechatgateway.csv', self.sampling_size)    # 2135
        
        print('Reading File transfer related csv files...')
        FILE_TRANSFER_gate_FTP_transfer = self.read_csv_file(r'FILE-TRANSFER_gate_FTP_transfer.csv', self.sampling_size)    # 2952252
        FILE_TRANSFER_gate_SFTP_filetransfer = self.read_csv_file(r'FILE-TRANSFER_gate_SFTP_filetransfer.csv', self.sampling_size)  # 929868
        FILE_TRANSFER_tor_skype_transfer = self.read_csv_file(r'FILE-TRANSFER_tor_skype_transfer.csv', self.sampling_size)  # 123862
        
        print('Reading Mail related csv files...')
        MAIL_gate_Email_IMAP_filetransfer = self.read_csv_file(r'MAIL_gate_Email_IMAP_filetransfer.csv', self.sampling_size)    # 95748
        MAIL_gate_POP_filetransfer = self.read_csv_file(r'MAIL_gate_POP_filetransfer.csv', self.sampling_size)  # 77794
        MAIL_Gateway_Thunderbird_Imap = self.read_csv_file(r'MAIL_Gateway_Thunderbird_Imap.csv', self.sampling_size)    # 133752
        MAIL_Gateway_Thunderbird_POP = self.read_csv_file(r'MAIL_Gateway_Thunderbird_POP.csv', self.sampling_size)  # 84118
        
        print('Reading P2P related csv files...')
        P2P_tor_p2p_multipleSpeed = self.read_csv_file(r'P2P_tor_p2p_multipleSpeed.csv', self.sampling_size)    # 1342497
        P2P_tor_p2p_vuze = self.read_csv_file(r'P2P_tor_p2p_vuze.csv', self.sampling_size)  # 1448963
        
        print('Reading Video related csv files...')
        VIDEO_Vimeo_Gateway = self.read_csv_file(r'VIDEO_Vimeo_Gateway.csv', self.sampling_size)    # 1355043
        VIDEO_Youtube_Flash_Gateway = self.read_csv_file(r'VIDEO_Youtube_Flash_Gateway.csv', self.sampling_size)    # 502737
        VIDEO_Youtube_HTML5_Gateway = self.read_csv_file(r'VIDEO_Youtube_HTML5_Gateway.csv', self.sampling_size)    # 270387
        
        print('Reading VoIP related csv files...')
        VOIP_Facebook_Voice_Gateway = self.read_csv_file(r'VOIP_Facebook_Voice_Gateway.csv', self.sampling_size)    # 317194
        VOIP_gate_facebook_Audio = self.read_csv_file(r'VOIP_gate_facebook_Audio.csv', self.sampling_size)  # 220995
        VOIP_gate_hangout_audio = self.read_csv_file(r'VOIP_gate_hangout_audio.csv', self.sampling_size)    # 266986
        VOIP_gate_Skype_Audio = self.read_csv_file(r'VOIP_gate_Skype_Audio.csv', self.sampling_size)    # 213763
        VOIP_Hangouts_voice_Gateway = self.read_csv_file(r'VOIP_Hangouts_voice_Gateway.csv', self.sampling_size)    # 312869
        VOIP_Skype_Voice_Gateway = self.read_csv_file(r'VOIP_Skype_Voice_Gateway.csv', self.sampling_size)  # 217814

        print('Reading nonTor csv files...')
        print('Reading Audio related csv files...')
        facebook_Audio = self.read_csv_file(r'facebook_Audio.csv', self.sampling_size)   # 218800
        Hangout_Audio = self.read_csv_file(r'Hangout_Audio.csv', self.sampling_size) # 244039
        Skype_Audio = self.read_csv_file(r'Skype_Audio.csv', self.sampling_size) # 199036
        spotify = self.read_csv_file(r'spotify.csv', self.sampling_size) # 42947
        spotify2 = self.read_csv_file(r'spotify2.csv', self.sampling_size)   # 165372
        spotifyAndrew = self.read_csv_file(r'spotifyAndrew.csv', self.sampling_size) # 34480
       
        print('Reading Browsing related csv files...')
        browsing = self.read_csv_file(r'browsing.csv', self.sampling_size)   # 219535
        browsing_ara = self.read_csv_file(r'browsing_ara.csv', self.sampling_size)   # 118384
        browsing_ara2 = self.read_csv_file(r'browsing_ara2.csv', self.sampling_size) # 88520
        browsing_ger = self.read_csv_file(r'browsing_ger.csv', self.sampling_size)   # 185055
        browsing2 = self.read_csv_file(r'browsing2.csv', self.sampling_size) # 54933
        ssl = self.read_csv_file(r'ssl.csv', self.sampling_size) # 52855
        SSL_Browsing = self.read_csv_file(r'SSL_Browsing.csv', self.sampling_size)   # 97926
        
        print('Reading Chat related csv files...')
        AIM_Chat = self.read_csv_file(r'AIM_Chat.csv', self.sampling_size)   # 526
        aimchat = self.read_csv_file(r'aimchat.csv', self.sampling_size) # 190
        facebook_chat = self.read_csv_file(r'facebook_chat.csv', self.sampling_size) # 2969
        facebookchat = self.read_csv_file(r'facebookchat.csv', self.sampling_size)   # 5814
        hangout_chat = self.read_csv_file(r'hangout_chat.csv', self.sampling_size)   # 3510
        hangoutschat = self.read_csv_file(r'hangoutschat.csv', self.sampling_size)   # 2994
        ICQ_Chat = self.read_csv_file(r'ICQ_Chat.csv', self.sampling_size)   # 386
        icqchat = self.read_csv_file(r'icqchat.csv', self.sampling_size) # 532
        skype_chat = self.read_csv_file(r'skype_chat.csv', self.sampling_size)   # 2121
        skypechat = self.read_csv_file(r'skypechat.csv', self.sampling_size) # 2415

        print('Reading File transfer related csv files...')
        FTP_filetransfer = self.read_csv_file(r'FTP_filetransfer.csv', self.sampling_size)   # 1710680
        SFTP_filetransfer = self.read_csv_file(r'SFTP_filetransfer.csv', self.sampling_size) # 526882
        skype_transfer = self.read_csv_file(r'skype_transfer.csv', self.sampling_size)   # 143083

        print('Reading Mail related csv files...')
        Email_IMAP_filetransfer = self.read_csv_file(r'Email_IMAP_filetransfer.csv', self.sampling_size) # 71235
        POP_filetransfer = self.read_csv_file(r'POP_filetransfer.csv', self.sampling_size)   # 58913
        Workstation_Thunderbird_Imap = self.read_csv_file(r'Workstation_Thunderbird_Imap.csv', self.sampling_size)   # 81653
        Workstation_Thunderbird_POP = self.read_csv_file(r'Workstation_Thunderbird_POP.csv', self.sampling_size) # 113529
        
        print('Reading P2P related csv files...')
        p2p_multipleSpeed = self.read_csv_file(r'p2p_multipleSpeed.csv', self.sampling_size) # 1328955
        p2p_vuze = self.read_csv_file(r'p2p_vuze.csv', self.sampling_size)   # 1109575

        print('Reading Video related csv files...')
        Vimeo_Workstation = self.read_csv_file(r'Vimeo_Workstation.csv', self.sampling_size) # 984223
        Youtube_Flash_Workstation = self.read_csv_file(r'Youtube_Flash_Workstation.csv', self.sampling_size) # 494769
        Youtube_HTML5_Workstation = self.read_csv_file(r'Youtube_HTML5_Workstation.csv', self.sampling_size) # 265313

        print('Reading VoIP related csv files...')
        Facebook_Voice_Workstation = self.read_csv_file(r'Facebook_Voice_Workstation.csv', self.sampling_size)   # 253691
        Hangouts_voice_Workstation = self.read_csv_file(r'Hangouts_voice_Workstation.csv', self.sampling_size)   # 261894
        Skype_Voice_Workstation = self.read_csv_file(r'Skype_Voice_Workstation.csv', self.sampling_size) # 194319

        
        # 2.Split datasets into the training and testing sets (Shuffle is True by default)
        print('Processing Tor files... (train/test split + object concatenation)')
        AUDIO_spotifygateway_train, AUDIO_spotifygateway_test = train_test_split(AUDIO_spotifygateway, test_size=self.test_size)
        AUDIO_tor_spotify_train, AUDIO_tor_spotify_test = train_test_split(AUDIO_tor_spotify, test_size=self.test_size)
        AUDIO_tor_spotify2_train, AUDIO_tor_spotify2_test = train_test_split(AUDIO_tor_spotify2, test_size=self.test_size)
        BROWSING_gate_SSL_Browsing_train, BROWSING_gate_SSL_Browsing_test = train_test_split(BROWSING_gate_SSL_Browsing, test_size=self.test_size)
        BROWSING_ssl_browsing_gateway_train, BROWSING_ssl_browsing_gateway_test = train_test_split(BROWSING_ssl_browsing_gateway, test_size=self.test_size)
        BROWSING_tor_browsing_ara_train, BROWSING_tor_browsing_ara_test = train_test_split(BROWSING_tor_browsing_ara, test_size=self.test_size)
        BROWSING_tor_browsing_ger_train, BROWSING_tor_browsing_ger_test = train_test_split(BROWSING_tor_browsing_ger, test_size=self.test_size)
        BROWSING_tor_browsing_mam_train, BROWSING_tor_browsing_mam_test = train_test_split(BROWSING_tor_browsing_mam, test_size=self.test_size)
        BROWSING_tor_browsing_mam2_train, BROWSING_tor_browsing_mam2_test = train_test_split(BROWSING_tor_browsing_mam2, test_size=self.test_size)
        CHAT_aimchatgateway_train, CHAT_aimchatgateway_test = train_test_split(CHAT_aimchatgateway, test_size=self.test_size)
        CHAT_facebookchatgateway_train, CHAT_facebookchatgateway_test = train_test_split(CHAT_facebookchatgateway, test_size=self.test_size)
        CHAT_gate_AIM_chat_train, CHAT_gate_AIM_chat_test = train_test_split(CHAT_gate_AIM_chat, test_size=self.test_size)
        CHAT_gate_facebook_chat_train, CHAT_gate_facebook_chat_test = train_test_split(CHAT_gate_facebook_chat, test_size=self.test_size)
        CHAT_gate_hangout_chat_train, CHAT_gate_hangout_chat_test = train_test_split(CHAT_gate_hangout_chat, test_size=self.test_size)
        CHAT_gate_ICQ_chat_train, CHAT_gate_ICQ_chat_test = train_test_split(CHAT_gate_ICQ_chat, test_size=self.test_size)
        CHAT_gate_skype_chat_train, CHAT_gate_skype_chat_test = train_test_split(CHAT_gate_skype_chat, test_size=self.test_size)
        CHAT_hangoutschatgateway_train, CHAT_hangoutschatgateway_test = train_test_split(CHAT_hangoutschatgateway, test_size=self.test_size)
        CHAT_icqchatgateway_train, CHAT_icqchatgateway_test = train_test_split(CHAT_icqchatgateway, test_size=self.test_size)
        CHAT_skypechatgateway_train, CHAT_skypechatgateway_test = train_test_split(CHAT_skypechatgateway, test_size=self.test_size)
        FILE_TRANSFER_gate_FTP_transfer_train, FILE_TRANSFER_gate_FTP_transfer_test = train_test_split(FILE_TRANSFER_gate_FTP_transfer, test_size=self.test_size)
        FILE_TRANSFER_gate_SFTP_filetransfer_train, FILE_TRANSFER_gate_SFTP_filetransfer_test = train_test_split(FILE_TRANSFER_gate_SFTP_filetransfer, test_size=self.test_size)
        FILE_TRANSFER_tor_skype_transfer_train, FILE_TRANSFER_tor_skype_transfer_test = train_test_split(FILE_TRANSFER_tor_skype_transfer, test_size=self.test_size)
        MAIL_gate_Email_IMAP_filetransfer_train, MAIL_gate_Email_IMAP_filetransfer_test = train_test_split(MAIL_gate_Email_IMAP_filetransfer, test_size=self.test_size)
        MAIL_gate_POP_filetransfer_train, MAIL_gate_POP_filetransfer_test = train_test_split(MAIL_gate_POP_filetransfer, test_size=self.test_size)
        MAIL_Gateway_Thunderbird_Imap_train, MAIL_Gateway_Thunderbird_Imap_test = train_test_split(MAIL_Gateway_Thunderbird_Imap, test_size=self.test_size)
        MAIL_Gateway_Thunderbird_POP_train, MAIL_Gateway_Thunderbird_POP_test = train_test_split(MAIL_Gateway_Thunderbird_POP, test_size=self.test_size)
        P2P_tor_p2p_multipleSpeed_train, P2P_tor_p2p_multipleSpeed_test = train_test_split(P2P_tor_p2p_multipleSpeed, test_size=self.test_size)
        P2P_tor_p2p_vuze_train, P2P_tor_p2p_vuze_test = train_test_split(P2P_tor_p2p_vuze, test_size=self.test_size)
        VIDEO_Vimeo_Gateway_train, VIDEO_Vimeo_Gateway_test = train_test_split(VIDEO_Vimeo_Gateway, test_size=self.test_size)
        VIDEO_Youtube_Flash_Gateway_train, VIDEO_Youtube_Flash_Gateway_test = train_test_split(VIDEO_Youtube_Flash_Gateway, test_size=self.test_size)
        VIDEO_Youtube_HTML5_Gateway_train, VIDEO_Youtube_HTML5_Gateway_test = train_test_split(VIDEO_Youtube_HTML5_Gateway, test_size=self.test_size)
        VOIP_Facebook_Voice_Gateway_train, VOIP_Facebook_Voice_Gateway_test = train_test_split(VOIP_Facebook_Voice_Gateway, test_size=self.test_size)
        VOIP_gate_facebook_Audio_train, VOIP_gate_facebook_Audio_test = train_test_split(VOIP_gate_facebook_Audio, test_size=self.test_size)
        VOIP_gate_hangout_audio_train, VOIP_gate_hangout_audio_test = train_test_split(VOIP_gate_hangout_audio, test_size=self.test_size)
        VOIP_gate_Skype_Audio_train, VOIP_gate_Skype_Audio_test = train_test_split(VOIP_gate_Skype_Audio, test_size=self.test_size)
        VOIP_Hangouts_voice_Gateway_train, VOIP_Hangouts_voice_Gateway_test = train_test_split(VOIP_Hangouts_voice_Gateway, test_size=self.test_size)
        VOIP_Skype_Voice_Gateway_train, VOIP_Skype_Voice_Gateway_test = train_test_split(VOIP_Skype_Voice_Gateway, test_size=self.test_size)

        print('Processing nonTor files... (train/test split + object concatenation)')
        AIM_Chat_train, AIM_Chat_test = train_test_split(AIM_Chat, test_size=self.test_size)
        aimchat_train, aimchat_test = train_test_split(aimchat, test_size=self.test_size)
        browsing_train, browsing_test = train_test_split(browsing, test_size=self.test_size)
        browsing_ara_train, browsing_ara_test = train_test_split(browsing_ara, test_size=self.test_size)
        browsing_ara2_train, browsing_ara2_test = train_test_split(browsing_ara2, test_size=self.test_size)
        browsing_ger_train, browsing_ger_test = train_test_split(browsing_ger, test_size=self.test_size)
        browsing2_train, browsing2_test = train_test_split(browsing2, test_size=self.test_size)
        Email_IMAP_filetransfer_train, Email_IMAP_filetransfer_test = train_test_split(Email_IMAP_filetransfer, test_size=self.test_size)
        facebook_Audio_train, facebook_Audio_test = train_test_split(facebook_Audio, test_size=self.test_size)
        facebook_chat_train, facebook_chat_test = train_test_split(facebook_chat, test_size=self.test_size)
        Facebook_Voice_Workstation_train, Facebook_Voice_Workstation_test = train_test_split(Facebook_Voice_Workstation, test_size=self.test_size)
        facebookchat_train, facebookchat_test = train_test_split(facebookchat, test_size=self.test_size)
        FTP_filetransfer_train, FTP_filetransfer_test = train_test_split(FTP_filetransfer, test_size=self.test_size)
        Hangout_Audio_train, Hangout_Audio_test = train_test_split(Hangout_Audio, test_size=self.test_size)
        hangout_chat_train, hangout_chat_test = train_test_split(hangout_chat, test_size=self.test_size)
        Hangouts_voice_Workstation_train, Hangouts_voice_Workstation_test = train_test_split(Hangouts_voice_Workstation, test_size=self.test_size)
        hangoutschat_train, hangoutschat_test = train_test_split(hangoutschat, test_size=self.test_size)
        ICQ_Chat_train, ICQ_Chat_test = train_test_split(ICQ_Chat, test_size=self.test_size)
        icqchat_train, icqchat_test = train_test_split(icqchat, test_size=self.test_size)
        p2p_multipleSpeed_train, p2p_multipleSpeed_test = train_test_split(p2p_multipleSpeed, test_size=self.test_size)
        p2p_vuze_train, p2p_vuze_test = train_test_split(p2p_vuze, test_size=self.test_size)
        POP_filetransfer_train, POP_filetransfer_test = train_test_split(POP_filetransfer, test_size=self.test_size)
        SFTP_filetransfer_train, SFTP_filetransfer_test = train_test_split(SFTP_filetransfer, test_size=self.test_size)
        Skype_Audio_train, Skype_Audio_test = train_test_split(Skype_Audio, test_size=self.test_size)
        skype_chat_train, skype_chat_test = train_test_split(skype_chat, test_size=self.test_size)
        skype_transfer_train, skype_transfer_test = train_test_split(skype_transfer, test_size=self.test_size)
        Skype_Voice_Workstation_train, Skype_Voice_Workstation_test = train_test_split(Skype_Voice_Workstation, test_size=self.test_size)
        skypechat_train, skypechat_test = train_test_split(skypechat, test_size=self.test_size)
        spotify_train, spotify_test = train_test_split(spotify, test_size=self.test_size)
        spotify2_train, spotify2_test = train_test_split(spotify2, test_size=self.test_size)
        spotifyAndrew_train, spotifyAndrew_test = train_test_split(spotifyAndrew, test_size=self.test_size)
        ssl_train, ssl_test = train_test_split(ssl, test_size=self.test_size)
        SSL_Browsing_train, SSL_Browsing_test = train_test_split(SSL_Browsing, test_size=self.test_size)
        Vimeo_Workstation_train, Vimeo_Workstation_test = train_test_split(Vimeo_Workstation, test_size=self.test_size)
        Workstation_Thunderbird_Imap_train, Workstation_Thunderbird_Imap_test = train_test_split(Workstation_Thunderbird_Imap, test_size=self.test_size)
        Workstation_Thunderbird_POP_train, Workstation_Thunderbird_POP_test = train_test_split(Workstation_Thunderbird_POP, test_size=self.test_size)
        Youtube_Flash_Workstation_train, Youtube_Flash_Workstation_test = train_test_split(Youtube_Flash_Workstation, test_size=self.test_size)
        Youtube_HTML5_Workstation_train, Youtube_HTML5_Workstation_test = train_test_split(Youtube_HTML5_Workstation, test_size=self.test_size)

        # 3.Merge training and testing sets respectively.
        concatenated_tor_train_set = np.concatenate((AUDIO_spotifygateway_train, AUDIO_tor_spotify_train, AUDIO_tor_spotify2_train, BROWSING_gate_SSL_Browsing_train, BROWSING_ssl_browsing_gateway_train
                                                , BROWSING_tor_browsing_ara_train, BROWSING_tor_browsing_ger_train, BROWSING_tor_browsing_mam_train, BROWSING_tor_browsing_mam2_train, CHAT_aimchatgateway_train
                                                , CHAT_facebookchatgateway_train, CHAT_gate_AIM_chat_train, CHAT_gate_facebook_chat_train, CHAT_gate_hangout_chat_train, CHAT_gate_ICQ_chat_train
                                                , CHAT_gate_skype_chat_train, CHAT_hangoutschatgateway_train, CHAT_icqchatgateway_train, CHAT_skypechatgateway_train, FILE_TRANSFER_gate_FTP_transfer_train
                                                , FILE_TRANSFER_gate_SFTP_filetransfer_train, FILE_TRANSFER_tor_skype_transfer_train, MAIL_gate_Email_IMAP_filetransfer_train, MAIL_gate_POP_filetransfer_train, MAIL_Gateway_Thunderbird_Imap_train
                                                , MAIL_Gateway_Thunderbird_POP_train, P2P_tor_p2p_multipleSpeed_train, P2P_tor_p2p_vuze_train, VIDEO_Vimeo_Gateway_train, VIDEO_Youtube_Flash_Gateway_train
                                                , VIDEO_Youtube_HTML5_Gateway_train, VOIP_Facebook_Voice_Gateway_train, VOIP_gate_facebook_Audio_train, VOIP_gate_hangout_audio_train, VOIP_gate_Skype_Audio_train
                                                , VOIP_Hangouts_voice_Gateway_train, VOIP_Skype_Voice_Gateway_train), axis=0)
        
        concatenated_tor_test_set = np.concatenate((AUDIO_spotifygateway_test, AUDIO_tor_spotify_test, AUDIO_tor_spotify2_test, BROWSING_gate_SSL_Browsing_test, BROWSING_ssl_browsing_gateway_test
                                                , BROWSING_tor_browsing_ara_test, BROWSING_tor_browsing_ger_test, BROWSING_tor_browsing_mam_test, BROWSING_tor_browsing_mam2_test, CHAT_aimchatgateway_test
                                                , CHAT_facebookchatgateway_test, CHAT_gate_AIM_chat_test, CHAT_gate_facebook_chat_test, CHAT_gate_hangout_chat_test, CHAT_gate_ICQ_chat_test
                                                , CHAT_gate_skype_chat_test, CHAT_hangoutschatgateway_test, CHAT_icqchatgateway_test, CHAT_skypechatgateway_test, FILE_TRANSFER_gate_FTP_transfer_test
                                                , FILE_TRANSFER_gate_SFTP_filetransfer_test, FILE_TRANSFER_tor_skype_transfer_test, MAIL_gate_Email_IMAP_filetransfer_test, MAIL_gate_POP_filetransfer_test, MAIL_Gateway_Thunderbird_Imap_test
                                                , MAIL_Gateway_Thunderbird_POP_test, P2P_tor_p2p_multipleSpeed_test, P2P_tor_p2p_vuze_test, VIDEO_Vimeo_Gateway_test, VIDEO_Youtube_Flash_Gateway_test
                                                , VIDEO_Youtube_HTML5_Gateway_test, VOIP_Facebook_Voice_Gateway_test, VOIP_gate_facebook_Audio_test, VOIP_gate_hangout_audio_test, VOIP_gate_Skype_Audio_test
                                                , VOIP_Hangouts_voice_Gateway_test, VOIP_Skype_Voice_Gateway_test), axis=0)

        concatenated_nonTor_train_set = np.concatenate((AIM_Chat_train, aimchat_train, browsing_train, browsing_ara_train, browsing_ara2_train
                                                , browsing_ger_train, browsing2_train, Email_IMAP_filetransfer_train, facebook_Audio_train, facebook_chat_train
                                                , Facebook_Voice_Workstation_train, facebookchat_train, FTP_filetransfer_train, Hangout_Audio_train, hangout_chat_train
                                                , Hangouts_voice_Workstation_train, hangoutschat_train, ICQ_Chat_train, icqchat_train, p2p_multipleSpeed_train
                                                , p2p_vuze_train, POP_filetransfer_train, SFTP_filetransfer_train, Skype_Audio_train, skype_chat_train
                                                , skype_transfer_train, Skype_Voice_Workstation_train, skypechat_train, spotify_train, spotify2_train
                                                , spotifyAndrew_train, ssl_train, SSL_Browsing_train, Vimeo_Workstation_train, Workstation_Thunderbird_Imap_train
                                                , Workstation_Thunderbird_POP_train, Youtube_Flash_Workstation_train, Youtube_HTML5_Workstation_train), axis=0)

        concatenated_nonTor_test_set = np.concatenate((AIM_Chat_test, aimchat_test, browsing_test, browsing_ara_test, browsing_ara2_test
                                                , browsing_ger_test, browsing2_test, Email_IMAP_filetransfer_test, facebook_Audio_test, facebook_chat_test
                                                , Facebook_Voice_Workstation_test, facebookchat_test, FTP_filetransfer_test, Hangout_Audio_test, hangout_chat_test
                                                , Hangouts_voice_Workstation_test, hangoutschat_test, ICQ_Chat_test, icqchat_test, p2p_multipleSpeed_test
                                                , p2p_vuze_test, POP_filetransfer_test, SFTP_filetransfer_test, Skype_Audio_test, skype_chat_test
                                                , skype_transfer_test, Skype_Voice_Workstation_test, skypechat_test, spotify_test, spotify2_test
                                                , spotifyAndrew_test, ssl_test, SSL_Browsing_test, Vimeo_Workstation_test, Workstation_Thunderbird_Imap_test
                                                , Workstation_Thunderbird_POP_test, Youtube_Flash_Workstation_test, Youtube_HTML5_Workstation_test), axis=0)

        concatenated_train_set = np.concatenate((concatenated_tor_train_set, concatenated_nonTor_train_set), axis=0)
        concatenated_test_set = np.concatenate((concatenated_tor_test_set, concatenated_nonTor_test_set), axis=0)

        concatenated_train_set = shuffle(concatenated_train_set)
        concatenated_test_set = shuffle(concatenated_test_set)

        self.x_train = concatenated_train_set[:, 0:-1]
        self.y_train = concatenated_train_set[:, [-1]] # 0 ~ 15
        self.train_length = self.x_train.shape[0]

        self.x_test = concatenated_test_set[:, 0:-1]
        self.y_test = concatenated_test_set[:, [-1]]
        self.test_length = self.x_test.shape[0]

#ds = Dataset('123', 10, 0.1)
