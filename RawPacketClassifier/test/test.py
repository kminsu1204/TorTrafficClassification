import numpy as np
import pandas as pd
import csv
import sys

#print(csv.field_size_limit(sys.maxsize))
#import pdb;pdb.set_trace()
#nonTor_Labeled = pd.read_csv(r'nonTor_Labeled/Workstation_Thunderbird_POP_labeled.csv', delimiter=',', dtype=np.float32, header=None)
#nonTor_Labeled =[]
csv_delimiter = '\0'
csv.field_size_limit(100000000)

#with open(r'nonTor_Labeled/ICQ_Chat_labeled.csv', 'r') as csvfile:
#    csvreader = csv.reader(csvfile, delimiter=',')
#    nonTor_Labeled = list(csvreader) 
    #for row in csvreader:
    #    nonTor_Labeled.append(row) 
    #return nonTor_Labeled

#print (nonTor_Labeled[[49908]])

print('11...')
AIM_Chat = pd.read_csv(r'nonTor/AIM_Chat.csv', delimiter=',', dtype=np.float32, header=None, error_bad_lines=False)   # 526
print('12...')
aimchat = pd.read_csv(r'nonTor/aimchat.csv', delimiter=',', dtype=np.float32, header=None, error_bad_lines=False) # 190
print('13...')
facebook_chat = pd.read_csv(r'nonTor/facebook_chat.csv', delimiter=',', dtype=np.float32, header=None, error_bad_lines=False) # 2969
print('14...')
facebookchat = pd.read_csv(r'nonTor/facebookchat.csv', delimiter=',', dtype=np.float32, header=None, error_bad_lines=False)   # 5814
print('15...')
hangout_chat = pd.read_csv(r'nonTor/hangout_chat.csv', delimiter=',', dtype=np.float32, header=None, error_bad_lines=False)   # 3510
print('16...')
hangoutschat = pd.read_csv(r'nonTor/hangoutschat.csv', delimiter=',', dtype=np.float32, header=None, error_bad_lines=False)   # 2994
print('17...')
ICQ_Chat = pd.read_csv(r'nonTor/ICQ_Chat.csv', delimiter=',', dtype=np.float32, header=None, error_bad_lines=False)   # 386
print('18...')
icqchat = pd.read_csv(r'nonTor/icqchat.csv', delimiter=',', dtype=np.float32, header=None, error_bad_lines=False) # 532
print('19...')
skype_chat = pd.read_csv(r'nonTor/skype_chat.csv', delimiter=',', dtype=np.float32, header=None, error_bad_lines=False)   # 2121
print('20...')
skypechat = pd.read_csv(r'nonTor/skypechat.csv', delimiter=',', dtype=np.float32, header=None, error_bad_lines=False) # 2415


