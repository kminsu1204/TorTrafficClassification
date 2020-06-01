ECHO OFF

cd Pcaps\nonTor

tshark -r browsing.pcap -x > browsing.txt
tshark -r browsing_ara.pcap -x > browsing_ara.txt
tshark -r browsing_ara2.pcap -x > browsing_ara2.txt
tshark -r browsing_ger.pcap -x > browsing_ger.txt
tshark -r browsing2.pcap -x > browsing2.txt
tshark -r browsing2-1.pcap -x > browsing2-1.txt
tshark -r browsing2-2.pcap -x > browsing2-2.txt
tshark -r Email_IMAP_filetransfer.pcap -x > Email_IMAP_filetransfer.txt
tshark -r facebook_Audio.pcap -x > facebook_Audio.txt
tshark -r facebook_chat.pcap -x > facebook_chat.txt

tshark -r Facebook_Voice_Workstation.pcap -x > Facebook_Voice_Workstation.txt
tshark -r facebookchat.pcap -x > facebookchat.txt
tshark -r FTP_filetransfer.pcap -x > FTP_filetransfer.txt
tshark -r Hangout_Audio.pcap -x > Hangout_Audio.txt
tshark -r hangout_chat.pcap -x > hangout_chat.txt
tshark -r Hangouts_voice_Workstation.pcap -x > Hangouts_voice_Workstation.txt
tshark -r hangoutschat.pcap -x > hangoutschat.txt
tshark -r p2p_multipleSpeed.pcap -x > p2p_multipleSpeed.txt
tshark -r p2p_multipleSpeed2-1.pcap -x > p2p_multipleSpeed2-1.txt
tshark -r p2p_vuze.pcap -x > p2p_vuze.txt

tshark -r p2p_vuze2-1.pcap -x > p2p_vuze2-1.txt
tshark -r POP_filetransfer.pcap -x > POP_filetransfer.txt
tshark -r SFTP_filetransfer.pcap -x > SFTP_filetransfer.txt
tshark -r Skype_Audio.pcap -x > Skype_Audio.txt
tshark -r skype_chat.pcap -x > skype_chat.txt
tshark -r skype_transfer.pcap -x > skype_transfer.txt
tshark -r Skype_Voice_Workstation.pcap -x > Skype_Voice_Workstation.txt
tshark -r skypechat.pcap -x > skypechat.txt
tshark -r spotify.pcap -x > spotify.txt
tshark -r spotify2.pcap -x > spotify2.txt

tshark -r spotify2-1.pcap -x > spotify2-1.txt
tshark -r spotify2-2.pcap -x > spotify2-2.txt
tshark -r spotifyAndrew.pcap -x > spotifyAndrew.txt
tshark -r ssl.pcap -x > ssl.txt
tshark -r SSL_Browsing.pcap -x > SSL_Browsing.txt
tshark -r Vimeo_Workstation.pcap -x > Vimeo_Workstation.txt
tshark -r Workstation_Thunderbird_Imap.pcap -x > Workstation_Thunderbird_Imap.txt
tshark -r Workstation_Thunderbird_POP.pcap -x > Workstation_Thunderbird_POP.txt
tshark -r Youtube_Flash_Workstation.pcap -x > Youtube_Flash_Workstation.txt
tshark -r Youtube_HTML5_Workstation.pcap -x > Youtube_HTML5_Workstation.txt

PAUSE