ECHO OFF

cd Pcaps\tor

tshark -r AUDIO_spotifygateway.pcap -x > AUDIO_spotifygateway.txt
tshark -r AUDIO_tor_spotify.pcap -x > AUDIO_tor_spotify.txt
tshark -r AUDIO_tor_spotify2.pcap -x > AUDIO_tor_spotify2.txt
tshark -r BROWSING_gate_SSL_Browsing.pcap -x > BROWSING_gate_SSL_Browsing.txt
tshark -r BROWSING_ssl_browsing_gateway.pcap -x > BROWSING_ssl_browsing_gateway.txt
tshark -r BROWSING_tor_browsing_ara.pcap -x > BROWSING_tor_browsing_ara.txt
tshark -r BROWSING_tor_browsing_ger.pcap -x > BROWSING_tor_browsing_ger.txt
tshark -r BROWSING_tor_browsing_mam.pcap -x > BROWSING_tor_browsing_mam.txt
tshark -r BROWSING_tor_browsing_mam2.pcap -x > BROWSING_tor_browsing_mam2.txt
tshark -r CHAT_aimchatgateway.pcap -x > CHAT_aimchatgateway.txt

tshark -r CHAT_facebookchatgateway.pcap -x > CHAT_facebookchatgateway.txt
tshark -r CHAT_gate_facebook_chat.pcap -x > CHAT_gate_facebook_chat.txt
tshark -r CHAT_gate_hangout_chat.pcap -x > CHAT_gate_hangout_chat.txt
tshark -r CHAT_gate_ICQ_chat.pcap -x > CHAT_gate_ICQ_chat.txt
tshark -r CHAT_gate_skype_chat.pcap -x > CHAT_gate_skype_chat.txt
tshark -r CHAT_hangoutschatgateway.pcap -x > CHAT_hangoutschatgateway.txt
tshark -r CHAT_icqchatgateway -x > CHAT_icqchatgateway.txt
tshark -r CHAT_skypechatgateway.pcap -x > CHAT_skypechatgateway.txt
tshark -r FILE-TRANSFER_gate_FTP_transfer.pcap -x > FILE-TRANSFER_gate_FTP_transfer.txt
tshark -r FILE-TRANSFER_tor_skype_transfer.pcap -x > FILE-TRANSFER_tor_skype_transfer.txt

tshark -r MAIL_gate_Email_IMAP_filetransfer.pcap -x > MAIL_gate_Email_IMAP_filetransfer.txt
tshark -r MAIL_gate_POP_filetransfer.pcap -x > MAIL_gate_POP_filetransfer.txt
tshark -r MAIL_Gateway_Thunderbird_Imap.pcap -x > MAIL_Gateway_Thunderbird_Imap.txt
tshark -r MAIL_Gateway_Thunderbird_POP.pcap -x > MAIL_Gateway_Thunderbird_POP.txt
tshark -r P2P_tor_p2p_vuze.pcap -x > P2P_tor_p2p_vuze.txt
tshark -r tor_p2p_multipleSpeed2-1.pcap -x > tor_p2p_multipleSpeed2-1.txt
tshark -r tor_p2p_vuze-2-1.pcap -x > tor_p2p_vuze-2-1.txt
tshark -r tor_spotify2-1.pcap -x > tor_spotify2-1.txt
tshark -r tor_spotify2-2.pcap -x > tor_spotify2-2.txt
tshark -r VIDEO_Vimeo_Gateway.pcap -x > VIDEO_Vimeo_Gateway.txt

tshark -r VIDEO_Youtube_Flash_Gateway.pcap -x > VIDEO_Youtube_Flash_Gateway.txt
tshark -r VIDEO_Youtube_HTML5_Gateway.pcap -x > VIDEO_Youtube_HTML5_Gateway.txt
tshark -r VOIP_Facebook_Voice_Gateway.pcap -x > VOIP_Facebook_Voice_Gateway.txt
tshark -r VOIP_gate_facebook_Audio.pcap -x > VOIP_gate_facebook_Audio.txt
tshark -r VOIP_gate_hangout_audio.pcap -x > VOIP_gate_hangout_audio.txt
tshark -r VOIP_gate_Skype_Audio.pcap -x > VOIP_gate_Skype_Audio.txt
tshark -r VOIP_Hangouts_voice_Gateway.pcap -x > VOIP_Hangouts_voice_Gateway.txt
tshark -r VOIP_Skype_Voice_Gateway.pcap -x > VOIP_Skype_Voice_Gateway.txt

PAUSE