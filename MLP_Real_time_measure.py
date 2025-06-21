import AIO
import motor
import sys
import os
import csv
from collections import deque
import pandas as pd
import numpy as np
import serial
import scipy
import collections
import datetime
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

##############################################################################
name='デモ'

CONTEC=b'AIO003'
COM_number='COM6'

num_of_joints  = 3
num_of_channels= 6  #計測チャンネル数

#######################################################################################################
buffer_size = 1600        #[Hz]
#######################################################################################################

MAV_buffer_size   = 800       #MAVの平滑幅 < 1600　(1600 => 1[s])
MAV_sampling_size = 100       #MAVのダウンサンプリングの幅 ex.100Hz
n = 128                       #STFTのパラメータ
fs = buffer_size              #STFTのパラメータ
shift = n//2                  #STFTのパラメータ
win = np.hamming(n)           #STFTのパラメータ
dim = 16                      #特徴次元数

###学習開始==========================================================================================================================
#計測周波数[Hz]
sampling_rate=1600.0 
#学習で信号を保存する時間[s](2[s]以上)
learning_time=3
#計測を打ち切るデータ数
limit_of_sampling=buffer_size*10*2*15

raw_data = []
signal_buffer = []
signal_buffer1 = []
emg=[]
MAV=[]
emgr=[]
MAVr=[]
x=[]
xr=[]
label=[]


cs=40960 #contec 初期値　+-2.5の時32767,0-5Vの時40960

for ch in range(0,num_of_channels):
    raw_data.append( deque([], maxlen=buffer_size*learning_time) ) 
    raw_data[ch].extend([cs]*buffer_size)
    signal_buffer.append( deque([], maxlen=buffer_size) ) 
    signal_buffer[ch].extend([cs]*buffer_size)
    signal_buffer1.append( deque([], maxlen=buffer_size) ) 
    signal_buffer1[ch].extend([cs]*buffer_size)

    emg.append( deque([], maxlen=buffer_size) )
    emg[ch].extend([0]*buffer_size)
    MAV.append( deque([], maxlen=buffer_size) )
    MAV[ch].extend([0]*buffer_size)
    
    emgr.append( deque([], maxlen=buffer_size) )
    emgr[ch].extend([0]*buffer_size)
    MAVr.append( deque([], maxlen=buffer_size) )
    MAVr[ch].extend([0]*buffer_size)
    xr.append(deque([]))

#デバイス名・IDを検索
device_name=AIO.queryAioDeviceName()

try:
    #デバイス名を直接指定する場合
    #aio=AIO.AIO(CONTEC)
    #queryAioDeviceNameで利用できるデバイス名を探しそれを使用する場合
    aio=AIO.AIO(device_name[0][0])
except ValueError as ve:
    print(ve)
    sys.exit()

#デバイスのリセット、ドライバの初期化
aio.resetDevice()

#アナログ入力ステータスをリセット
aio.resetAiStatus()

#デバイスメモリをリセット
aio.resetAiMemory()

#使用できるアナログ入力チャネルの最大数
print('MaxCh:%d'%aio.getAiMaxChannels())

#変換に使用するアナログ入力チャネル数の設定: Ex)8ch
aio.setAiChannels(num_of_channels)
print('UseCh:%d'%aio.getAiChannels())

#データ格納用メモリ形式をFIFOに設定
aio.setAiMemoryType(0)
print( "MemoryType:%d" % aio.getAiMemoryType())

#クロック種類の設定：内部
aio.setAiClockType(0)
print( "ClockType:%d" % aio.getAiClockType())

#変換速度の設定(µs): Ex)1600Hz->625µs
aio.setAiSamplingClock( 1000000.0/sampling_rate)
print( "SamplingClock:%f" % aio.getAiSamplingClock() )

#開始条件の設定：ソフトウェア
aio.setAiStartTrigger(0)
print( "StartTrigger:%d" % aio.getAiStartTrigger() )

#終了条件の設定：コマンド
aio.setAiStopTrigger(4)
print( "StopTrigger:%d" % aio.getAiStopTrigger() )

#変換スタート
#aio.resetAiMemory()
#aio.startAi()

#ser.reset_input_buffer()
#ser.reset_output_buffer()


print("Sampling start")

#メモリ内のサンプリング可能なデータ数を取得
#num_of_sampling=aio.getAiSamplingCount()

filename1 = 'リアルタイム学習/'+name+'生筋電.csv'
with open(filename1,'w',newline='') as f:
    pass

while True:
    count=0
    key=input('動作番号を入力:')
    if(key=="end"):
        print("学習データ計測終了")
        break
    try:
        prr=int(key)
        if(prr<0 or 26<prr):
            print("0~26かendを入力")
            continue
    except ValueError:
        print("0~26かendを入力")
        continue
    else:
        aio.resetAiMemory()
        aio.startAi()
        while (count<buffer_size*learning_time):
            num_of_sampling=aio.getAiSamplingCount()    
            try:
                if num_of_sampling > buffer_size :
                    num_of_sampling=buffer_size
                get_data=aio.getAiSamplingData(num_of_sampling)

            except ValueError as ve:
                print(ve)

            else:
                if(get_data[0]!=0):
                    for ch in range(num_of_channels):
                        raw_data[ch].extend(get_data[1][ch::num_of_channels])
                    count+=get_data[0]
        label=np.ones((buffer_size*learning_time,1))*prr
        semg=np.c_[raw_data[0],raw_data[1],raw_data[2],raw_data[3],raw_data[4],raw_data[5],label[:,0]]
        
        with open(filename1,'a',newline='') as f:
            writer = csv.writer(f)
            writer.writerows(semg)
        aio.stopAi()

reader = pd.read_csv(r"{}".format(filename1), header=None).values.tolist()
re = np.array(reader)

filename2 = 'リアルタイム学習/'+name+'特徴量.csv'
with open(filename2,'w',newline='') as f:
    pass
filename3 = 'リアルタイム学習/'+name+'ラベル.csv'
with open(filename3,'w',newline='') as f:
    pass
for k in range(len(re)//(buffer_size*learning_time)):
    x=[]
    ampAA=[]
    b=[]
    labelf=[]
    labelf=np.ones((MAV_sampling_size*(learning_time-1),1))*int(re[k*buffer_size*learning_time,num_of_channels])
    for ch in range(num_of_channels):
        #print("ch"+str(ch))
        x.append(deque([]))
        ampAA.append(deque([]))
        b.append(deque([]))
        for j in range((buffer_size*learning_time)*k+buffer_size//2,(buffer_size*learning_time)*(k+1)-buffer_size//2,buffer_size//MAV_sampling_size): #MAVのダウンサンプリングの幅 ex.16→1600/16=100Hz
            
            sb = re[j - MAV_buffer_size//2:j + MAV_buffer_size//2,ch]
            #emg[ch]=sb                                 #sigrecの時
            emg[ch] = 10 - (65535 - sb) / 65535 * 20    #sigrec以外の時
            ave1 = sum(emg[ch]) / len(emg[ch])
            MAV = sum(abs(emg[ch] - ave1)) / len(emg[ch])
            x[ch].append(MAV)
            f,t,Sxx=scipy.signal.spectrogram(emg[ch],fs=buffer_size,nfft=n,window=win,noverlap=shift)
            sum_sxx=[0.0]*dim
            sum_sxX=[0.0]*dim
            amp=[0.0]*dim
            for ii in range(dim):
                if ii==0 :
                    for i in range(int((n/2) * (1/(fs/2) * 5)), int((n/2) * (1/(fs/2) * 500/dim))):
                        sum_sxx[ii] += Sxx[i,:]
                else :
                    for i in range(int((n/2) * (1/(fs/2) * 500/dim)*ii), int((n/2) * (1/(fs/2) * (500/dim)*(ii+1)))):
                        sum_sxx[ii] += Sxx[i,:]
            for i in range (dim):
                sum_sxX[i]=sum(sum_sxx[i])/len(sum_sxx[i])
                amp[i]=(np.sqrt(sum_sxX[i]*(1/(n*(1/fs)))))*(np.sqrt(2))
                ampAA[ch].append([amp[i]])
        b[ch]=np.array(ampAA[ch]).reshape(MAV_sampling_size*(learning_time-1),dim)        
    feature=np.c_[b[0],b[1],b[2],b[3],b[4],b[5],x[0],x[1],x[2],x[3],x[4],x[5]]
    
    with open(filename2,'a',newline='') as f:
        writer = csv.writer(f)
        writer.writerows(feature)
    with open(filename3,'a',newline='') as f:
        writer = csv.writer(f)
        writer.writerows(labelf)
reader_x = pd.read_csv(r"{}".format(filename2), header=None).values.tolist()
re_x = np.array(reader_x)
reader_y = pd.read_csv(r"{}".format(filename3), header=None).values.tolist()
re_y = np.array(reader_y)
print("学習開始")
mlpc=MLPClassifier(hidden_layer_sizes=132,activation='logistic',learning_rate_init=0.001,tol=0.0001,random_state=1, max_iter=1000)
mlpc.fit(re_x, re_y)
pickle.dump(mlpc, open('リアルタイム学習/MLP_'+name+'.pkl', 'wb'))
print("学習終了")

"""
            for ch in range(0,num_of_channels):
                signal_buffer[ch].extend(get_data[1][ch::num_of_channels])
                signal_buffer1[ch]=get_data[1][ch::num_of_channels]
            label=[]
            label.append(deque([],maxlen=get_data[0]))
            label[0]=[[prr] for i in range (get_data[0])]
            print(get_data[0])
            print(label)
            
            
print("おわり")
#信号を保存するバッファを確保
#使用するChの数だけ，長さbuffer_sizeのキュー(待ち行列FIFO)を用意

files = os.listdir("./計測データ/{}/{}".format(name,processing_mode))
num_of_kinds=int(len(files))

if __name__ == '__main__':
    ser = serial.Serial(COM_number, 9600, timeout=1)



    
    ye=[]
    yw=[]
    yh=[]
    ampAA=[]
    bp=[]

    


    a=7 #0.5秒間のデータの個数

    bp.append( deque([], maxlen=a) )
    bp[0].extend([[0]]*a) 

    
    hk=motor.degree(gishuname)[0]  #4指の最大屈曲時のモータ指令値
    hs=motor.degree(gishuname)[1]  #4指の最大伸展時のモータ指令値
    wk=motor.degree(gishuname)[2]  #手首の最大回内時のモータ指令値
    ws=motor.degree(gishuname)[3]  #手首の最大回外時のモータ指令値
    ek=motor.degree(gishuname)[4]  #肘の最大屈曲時のモータ指令値
    es=motor.degree(gishuname)[5]  #肘の最大伸展時のモータ指令値
    tk=motor.degree(gishuname)[6]  #拇指の対立
    ts=motor.degree(gishuname)[7]  #拇指の並立

    ha=(hk+hs)/2
    ea=(ek+es)/2
    wa=(wk+ws)/2
    ta=tk

    yhds2=ha
    yeds=ea
    ywds=wa
    ytds=ta

    model=pickle.load(open('学習モデル/MLP_'+name+model_number+'.pkl','rb'))
###fit終了===========================================================================================================================   

    now=datetime.datetime.now()
###学習終了===========================================================================================================================   
    
    
   
    count=0

    
#   メインループ
    while True:

###リアルタイム推定開始===========================================================================================================================   
        
        
        
###特徴抽出開始===========================================================================================================================   

            ampAA=[]
            for ch in range(0,num_of_channels):
                signal_buffer[ch].extend(get_data[1][ch::num_of_channels])
                signal_buffer1[ch]=get_data[1][ch::num_of_channels]              

                sbr=np.array(signal_buffer[ch])
                emgr[ch]=10-(65535-sbr)/65535*20
                ave1=sum(emgr[ch])/len(signal_buffer[ch])
                MAVr[ch]=sum(abs(emgr[ch]-ave1))/len(emgr[ch])
                f,t,Sxx = scipy.signal.spectrogram(emgr[ch],fs=buffer_size,nfft=n,window=win,noverlap=shift)
                
                sum_sxx=[0.0]*dim
                sum_sxX=[0.0]*dim
                amp=[0.0]*dim
                for ii in range(dim):
                    if ii==0 :
                        for i in range(int((n/2) * (1/(fs/2) * 5)), int((n/2) * (1/(fs/2) * 500/dim))):
                            sum_sxx[ii] += Sxx[i,:]
                    else :
                        for i in range(int((n/2) * (1/(fs/2) * 500/dim)*ii), int((n/2) * (1/(fs/2) * (500/dim)*(ii+1)))):
                            sum_sxx[ii] += Sxx[i,:]
                for i in range (dim):
                    sum_sxX[i]=sum(sum_sxx[i])/len(sum_sxx[i])
                    amp[i]=(np.sqrt(sum_sxX[i]*(1/(n*(1/fs)))))*(np.sqrt(2))
                for i in range (dim):
                    ampAA.append([amp[i]])

                #emg[ch].extend([emg1])
                #MAV[ch].extend([MAV1]) 
###特徴抽出終了===========================================================================================================================   



            semg=np.c_[signal_buffer1[0],signal_buffer1[1],signal_buffer1[2],signal_buffer1[3],signal_buffer1[4],signal_buffer1[5]]

            filename2 = '[パターン6]semg' + now.strftime('%Y%m%d_%H%M%S') + '.csv'
            with open(filename2,'a',newline='') as f:
                writer = csv.writer(f)
        
                writer.writerows(semg)
            
            rea=[]
            b=np.array(ampAA).reshape(1,dim*(num_of_channels))

               
            xrr=np.c_[b,MAVr[0],MAVr[1],MAVr[2],MAVr[3],MAVr[4],MAVr[5]]
            xrr1=np.c_[MAVr[0],MAVr[1],MAVr[2],MAVr[3],MAVr[4],MAVr[5]]
            
            S1=np.c_[count/1600,1,xrr]

            
            filename1 = '[パターン6]MAV' + now.strftime('%Y%m%d_%H%M%S') + '.csv'
            with open(filename1,'a',newline='') as f:
               writer = csv.writer(f)
             
               writer.writerows(S1)

###筋電特徴から学習で作成したモデルを通して結果を推定開始===========================================================================================================================   

            o=model.predict_proba(xrr)
            #of = np.array(o)
            
            #np.set_printoptions(precision=3)
                  
            #print(of)
            #print(xrr)
            #y_pred = clf.predict(xrr)

            ss = 0.8 #閾値
            flag = 0
            for m, file in enumerate (files):
                if o[0][m]>ss:
                    y_pred=[m]
                    flag=1  
            if flag==0:
                y_pred=[len(files)]
###筋電特徴から学習で作成したモデルを通して結果を推定終了===========================================================================================================================   

###結果の平滑化開始===========================================================================================================================   

            bp[0].append(np.array(y_pred).tolist())
            bp1=np.array(bp[0])
            bp2=bp1.reshape(1,len(bp1))

            c=collections.Counter(bp2[0])
            prr=[c.most_common()[0][0]]
            #print(prr)
###結果の平滑化終了===========================================================================================================================   

###モータ指令値の計算開始===========================================================================================================================   

            dyh=(hk-hs)/(2*a*1)  #可動域を1sで移動
            dye=(ek-es)/(2*a*6)  #可動域を6sで移動
            dyw=(wk-ws)/(2*a*6)  #可動域を6sで移動
            dyt=(tk-ts)/(2*a*0.5)#可動域を0.5sで移動

            move=[]
            for m,file in enumerate(files):
                file=os.path.splitext(file)[0]
                move.append([])
                move[m].extend(motor.motormove(file))
            
            
            for m,file in enumerate(files):
                file=os.path.splitext(file)[0]
                if prr==[m]:
                    ytds  = ytds  + dyt*move[m][0]
                    yhds2 = yhds2 + dyh*move[m][0]
                    ywds  = ywds  + dye*move[m][1]
                    yeds  = yeds  + dyw*move[m][2]
                    print(file)
                    
            if prr==[len(files)]:
                print("識別不能(安静)")
                ytds  = ytds
                yhds2 = yhds2
                yeds  = yeds
                ywds  = ywds
                
            filename7 = '[パターン6]suiteikekka' + now.strftime('%Y%m%d_%H%M%S') + '.csv'
            with open(filename7,'a',newline='') as f:
               writer = csv.writer(f)
             
               writer.writerows([prr])

            
            if(hk<hs):#左手義手
                if (yhds2>hs):
                    yhds2=hs
                elif(yhds2<hk):
                    yhds2=hk
                if(ywds>ws):
                    ywds=ws
                elif(ywds<wk):
                    ywds=wk
                if(yeds>es):
                    yeds=es
                elif(yeds<ek):
                    yeds=ek
                if(ytds<ts):
                    ytds=ts
                elif(ytds>tk):
                    ytds=tk

            else:#右手義手
                if (yhds2<hs):
                    yhds2=hs
                elif(yhds2>hk):
                    yhds2=hk
                if(ywds<ws):
                    ywds=ws
                elif(ywds>wk):
                    ywds=wk
                if(yeds<es):
                    yeds=es
                elif(yeds>ek):
                    yeds=ek
                if(ytds>ts):
                    ytds=ts
                elif(ytds<tk):
                    ytds=tk

            #yhds2=np.clip(yhds2,hk,hs)
            #yeds=np.clip(yeds,ek,es)
            #ywds=np.clip(ywds,wk,ws)
            
            suideg=np.c_[yhds2 ,ywds, yeds ]
            ############print(suideg)
            filename5 = '[パターン6]cmd丸目前' + now.strftime('%Y%m%d_%H%M%S') + '.csv'
            with open(filename5,'a',newline='') as f:
                writer = csv.writer(f)
        
                writer.writerows(suideg)
     
###モータ指令値の計算終了===========================================================================================================================   


                    
            #cmd_header=b'MP'
            #cmd_data =[175,my_round_int(yhds2),150,150,my_round_int(yeds),my_round_int(ywds),150,150,150,150,150,150,150]

            cmd_data_1=str(int(ytds))+','+str(int(yhds2))+','+str(int(yeds))+','+str(int(ywds))
            #print(cmd_data_1)
            ser.write((cmd_data_1+'\r\n').encode('UTF-8'))
            #cmd_data=[母指,四指,-,-  , 肘,手首,- ,-  ,-  ,-  ,-  ,-  ,-  ]
            # バイトオブジェクトへの相互変換にto_bytes，from_bytesを使用する場合
            # 送信データの統合
            count+=1
            '''
            cmd=cmd_header+cmd_data[0].to_bytes(1,'little')+cmd_data[1].to_bytes(1,'little')+cmd_data[2].to_bytes(1,'little')+cmd_data[3].to_bytes(1,'little')+cmd_data[4].to_bytes(1,'little')+cmd_data[5].to_bytes(1,'little')+cmd_data[6].to_bytes(1,'little')+cmd_data[7].to_bytes(1,'little')+cmd_data[8].to_bytes(1,'little')+cmd_data[9].to_bytes(1,'little')+cmd_data[10].to_bytes(1,'little')+cmd_data[11].to_bytes(1,'little')+cmd_data[12].to_bytes(1,'little')
               
            cd=np.array([cmd_data])
            filename4 = '[パターン6]cmd' + now.strftime('%Y%m%d_%H%M%S') + '.csv'
            with open(filename4,'a',newline='') as f:
                writer = csv.writer(f)
        
                writer.writerows(cd)
            
            # データ送信
            ser.write(cmd)
            print(cmd)

            #print(cmd_data,count/1600,prr)

            count+=get_data[0]limit_of_sampling
            '''
"""        
""" #limit_of_samplin以上取得したら変換終了
        if count > limit_of_sampling:
            aio.stopAi()
            

            print("Sampling stop")

            ser.close() 
            break """
###リアルタイム推定終了===========================================================================================================================   
