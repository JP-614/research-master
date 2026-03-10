import serial
import time
import struct

class ManualMode():
    def __init__(self,COM_Number,baudrate=115200):
        self.ser= serial.Serial(COM_Number, baudrate=baudrate, timeout=0.1,write_timeout=None)
        #入出力バッファのクリア（念のため）
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        self.Date=b'20250818'

    def reset(self):
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        
    def clip_value(self,value, min_val, max_val):
        return max(min(value, max_val), min_val)
    
    def Send(self,SendData):
        # for i in range(len(SendData)):
        #     self.ser.write(SendData[i].to_bytes(1,'little',signed=False))
        self.ser.write(bytearray(SendData))
    def calcSum(self,SendData,index):
        ret = 0
        for i in range(index):
            ret += SendData[i]
        ret &= 0xff
        return ret

    # def CheckSum(message):
    # def MessageCheck():
    #     messages=ser.readlines()
    #     for i in range(len(messages)):
    #         if (len(messages[i]) % 8 != 0):
    #             messages[i]+=messages[i+1]
    #         print(len(message))
    def MessageCheck(self):
        messages=self.ser.readlines()
        #print(messages)
        for message in messages:
            if(message[1]==0x41):
                print('VER:',message[3:11])
                if(message[3:11]==self.Date):
                    print("バージョンが同じです")
                else:
                    print("バージョンが異なります")
            elif(message[1]==0x61):
                if(message[3]==0x13):
                    print('MODE:MANUAL')
            elif(message[1]==0x49):
                MotionNumber=message[2]
                MotorNumber=message[3]
                MiddleMotionNumber=message[4]
                self.MotorPosition=[]
                for motion in range(MotionNumber):
                    self.MotorPosition.append([])
                    for motor in range(MotorNumber):
                        self.MotorPosition[motion].append(message[6+MotorNumber*motion+motor])            
            elif(message[1]==0x6a):
                print('motor:',message[3],',degree:',message[4])
            else:
                print(messages)
                
    def MotorPosition_Humerus(self):
        return self.MotorPosition 
    
    def SendRequestVerInfo(self):
        SendData=[[] for i in range(8)]
        SendData[0]=0x01
        SendData[1]=0x05
        SendData[2]=0x02
        SendData[3]=0x41
        SendData[4]=0x00
        SendData[5]=0x03
        SendData[6]=self.calcSum(SendData,6)
        SendData[7]=0x0a
        self.Send(SendData)

    def SendMotorDegree(self,joint,degree):
        SendData=[[] for i in range(8)]
        SendData[0]=0x01
        SendData[1]=0x6a
        SendData[2]=0x02
        SendData[3]=joint
        SendData[4]=degree
        SendData[5]=0x03
        SendData[6]=self.calcSum(SendData,6)
        SendData[7]=0x0a
        self.Send(SendData)
        
    def SendModeCheck(self):
        SendData=[[] for i in range(8)]
        SendData[0]=0x01
        SendData[1]=0x05
        SendData[2]=0x02
        SendData[3]=0x61#Mode
        SendData[4]=0x00
        SendData[5]=0x03
        SendData[6]=self.calcSum(SendData,6)
        SendData[7]=0x0a
        self.Send(SendData)

    def SendMotorValueCheck(self): #ModeがSensorじゃないと送られてこない
        SendData=[[] for i in range(8)]
        SendData[0]=0x01
        SendData[1]=0x05
        SendData[2]=0x02
        SendData[3]=0x49 #Motor
        SendData[4]=0x00
        SendData[5]=0x03
        SendData[6]=self.calcSum(SendData,6)
        SendData[7]=0x0a
        self.Send(SendData)

    def SendModeChange(self,Mode):
        SendData=[[] for i in range(8)]
        SendData[0]=0x01
        SendData[1]=0x61
        SendData[2]=0x02
        if Mode =="Check":
            SendData[3]=0x01
        elif Mode == "Sensor":
            SendData[3]=0x02
        elif Mode == "Motion":
            SendData[3]=0x03
        elif Mode == "Manual":
            SendData[3]=0x13 
        SendData[4]=0x03
        SendData[5]=self.calcSum(SendData,5)
        SendData[6]=0x00
        SendData[7]=0x0a
        self.Send(SendData)

# Date='20241112'
# print(Date)
# SendVerInfo(Date)

if __name__ == '__main__':
    # Date=b'20250818'
    # #ポートオープン(COM3,baudrate=9600.タイムアウト0.1秒)
    #ser = serial.Serial('COM6', 115200, timeout=0.1)#COM18:SH
    # #発信側のcomポートを設定



    # #Arduinoとの通信では，通信の開通後と送信との間に2.5秒ほど、Arduinoのブートまで待たなければならない
    # time.sleep(2.5)
    # starttime=time.time()
    # ser.write(9223372036854775807)
    # finishtime=time.time()
    # shoritime=finishtime-starttime
    # print(shoritime)
    
    #入出力バッファのクリア（念のため）
    sh=ManualMode('COM6')
    sh.SendRequestVerInfo()
    sh.SendModeChange("Sensor")
    sh.SendMotorValueCheck()
    sh.MessageCheck()
    sh.SendModeChange("Manual")
    sh.SendModeCheck()
    while(True):
        # sh.MessageCheck()
        # print('関節')
        # joint=sh.clip_value(int(input()),0x00,0x0f)
        # print('角度')
        # degree=sh.clip_value(int(input()),0x00,0xff)
        joint=13
        degree=250
        starttime=time.time()
        sh.SendMotorDegree(joint,degree)
        finishtime=time.time()
        shoritime=finishtime-starttime
        print(shoritime)
        time.sleep(1)

        
    # ser.write(bytes('\x01\x05\x02\x41\x00\x03\x4c\n', encoding='utf-8', errors='replace'))#バージョン問い合わせ
    # ser.write(bytes('\x01\x61\x02\x13\x03\x7a\x00\n', encoding='utf-8', errors='replace'))#モード変更
    # ser.write(bytes('\x01\x05\x02\x61\x00\x03\x6c\n', encoding='utf-8', errors='replace'))#モード確認
    # #ser.write(bytes('\x01\x61\x02\x01\x03\x68\x00\n', encoding='utf-8', errors='replace'))#モード変更
    # time.sleep(5)
    # line=ser.readline()
    # sign=0x01
    # ser.write(sign.to_bytes(1,'little',signed=False))
    # sign=0x6a
    # ser.write(sign.to_bytes(1,'little',signed=False))
    # sign=0x02
    # ser.write(sign.to_bytes(1,'little',signed=False))
    # sign=0x0d
    # ser.write(sign.to_bytes(1,'little',signed=False))
    # sign=0xf2
    # ser.write(sign.to_bytes(1,'little',signed=False))
    # sign=0x03
    # ser.write(sign.to_bytes(1,'little',signed=False))
    # sign=0x6f
    # ser.write(sign.to_bytes(1,'little',signed=False))
    # sign=0x0a
    # ser.write(sign.to_bytes(1,'little',signed=False))
    # time.sleep(5)
    # sign=0x01
    # ser.write(sign.to_bytes(1,'little',signed=False))
    # sign=0x6a
    # ser.write(sign.to_bytes(1,'little',signed=False))
    # sign=0x02
    # ser.write(sign.to_bytes(1,'little',signed=False))
    # sign=0x0d
    # ser.write(sign.to_bytes(1,'little',signed=False))
    # sign=0x80
    # ser.write(sign.to_bytes(1,'little',signed=False))
    # sign=0x03
    # ser.write(sign.to_bytes(1,'little',signed=False))
    # sign=0xfd
    # ser.write(sign.to_bytes(1,'little',signed=False))
    # sign=0x0a
    # ser.write(sign.to_bytes(1,'little',signed=False))
    # # ser.write(0x6a)
    # # ser.write(0x02)
    # # ser.write(0x0f)
    # # ser.write(0x00)
    # # ser.write(0x03)
    # # ser.write(0x7f)
    # # ser.write(0x0a)
    # # #line_disp=line.strip().decode('UTF-8')
    # print(line)
    # while(True):
    #     #ser.write(bytes('\x01\x41\x02\x32\x30\x32\x33\x30\x32\x31\x35\x03\x16\x00\x00\x0a', encoding='utf-8', errors='replace'))
    #     #ser.write(bytes('\x01A\x0220230215\x03\x16\x00\x00\n', encoding='utf-8', errors='replace'))
    #     #ser.write(bytes('\x01\x6a\x02\x0d\xf0\x03\x6d\n', encoding='utf-8', errors='replace'))
    #     line=ser.readlines()
    #     #line_disp=line.strip().decode('UTF-8')
    #     print(line)
    #     time.sleep(5)
        

    #     #ser.write(bytes('\x01\x6a\x02\x0d\x01\x03\x7e\n', encoding='utf-8', errors='replace'))#動く
    #     print('拇指')
    #     time.sleep(5)
    #     line=ser.readlines()
    #     #line_disp=line.strip().decode('UTF-8')
    #     print(line)
    #     #ser.write(bytes('\x01\x6a\x02\x0c\x01\x03\x7d\n', encoding='utf-8', errors='replace'))#動く
    #     print('四指')
    #     time.sleep(5)
    #     line=ser.readlines()
    #     #line_disp=line.strip().decode('UTF-8')
    #     print(line)
    #     #ser.write(bytes('\x01\x6a\x02\x0f\x00\x03\x7f\n', encoding='utf-8', errors='replace'))#動く
    #     print('肘')
    #     time.sleep(5)
    #     line=ser.readlines()
    #     #line_disp=line.strip().decode('UTF-8')
    #     print(line)
    #     #ser.write(bytes('\x01\x6a\x02\x0e\x01\x03\x7f\n', encoding='utf-8', errors='replace'))#動く
    #     print('手首')
        
    #     # ser.write(bytes('\x01\x6a\x02\x0d\x03\x03\x80\n', encoding='utf-8', errors='replace'))#
    #     # ser.write(bytes('\x01\x6a\x02\x0c\x03\x03\x7f\n', encoding='utf-8', errors='replace'))#
    #     # ser.write(bytes('\x01\x6a\x02\x0f\x03\x03\x82\n', encoding='utf-8', errors='replace'))#
    #     # ser.write(bytes('\x01\x6a\x02\x0e\x03\x03\x81\n', encoding='utf-8', errors='replace'))#
        
    #     # ser.write(bytes('\x01\x6a\x02\x0d\x90\x03\x0d\n', encoding='utf-8', errors='replace'))
    #     # time.sleep(5)
    #     # ser.write(bytes('\x01\x6a\x02\x0c\x90\x03\x0c\n', encoding='utf-8', errors='replace'))
        
        
        
    #     # ser.write(bytes('\x01\x6a\x02\x0d\x02\x03\x7f\n', encoding='utf-8', errors='replace'))#
    #     # ser.write(bytes('\x01\x6a\x02\x0c\x03\x03\x7f\n', encoding='utf-8', errors='replace'))#
    #     # ser.write(bytes('\x01\x6a\x02\x0f\x03\x03\x82\n', encoding='utf-8', errors='replace'))#
    #     # ser.write(bytes('\x01\x6a\x02\x0e\x03\x03\x81\n', encoding='utf-8', errors='replace'))#
    #     #ser.write(bytes('\x01\x6a\x02\x0d\x0f\x03\x8c\n', encoding='utf-8', errors='replace'))
    #     #ser.write(bytes('\x01\x6a\x02\x0c\x0f\x03\x8b\n', encoding='utf-8', errors='replace'))
        
        
    # line=ser.readline()
    # #line_disp=line.strip().decode('UTF-8')
    # print(line)
    # sum=0x01+0x61+0x02+0x13+0x03
    # ser.write(bytes('0x010x610x020x130x03{:#04x}0x000x0a'.format(sum), encoding='utf-8', errors='replace'))
    # line=ser.readline()
    # print(line)
        
    # while(True):
    #     line=ser.readline()
    #     #line_disp=line.strip().decode('UTF-8')
    #     print(line)
    #     sum=0x01+0x6a+0x02+joint+deg+0x03
    #     degh=hex(joint)+hex(deg)
    #     #print(degh)
    #     print(bytes('0x010x6a0x02{:#04x}{:#04x}0x03{:#04x}0x000x000x000x000x000x000x0a'.format(joint,deg,sum), encoding='utf-8', errors='replace'))
    #     ser.write(bytes('0x010x6a0x02{:#04x}{:#04x}0x03{:#04x}0x000x000x000x000x000x000x0a'.format(joint,deg,sum), encoding='utf-8', errors='replace'))
    #     deg+=1
    #     if(deg>180):
    #         deg=1

    # # 1バイト送信　→　 1バイト受信
    
    # ser.write(b'F')
    # print(ser.read(1))

    # # 5バイト送信　→　 5バイト受信
    # ser.write(b'ABCDE')
    # print(ser.read(5))

    # # readlineのテスト
    # # readlineはdelimiter(区切り文字)である\nまで読み込む
    # ser.write(b'ABCDEF\nGHIJ\n')
    # print(ser.readline())
    # print(ser.readline())


    # # 2バイト整数を送信　→　 2バイト受信　→　2バイト整数として解釈
    # # 負の整数の場合，to_bytes，from_bytes共にsigned=Trueを指定する
    # # バイトオブジェクトへの相互変換にto_bytes，from_bytesを使用する場合
    # int_data=511
    # ser.write(int_data.to_bytes(2,'little'))
    # print( int.from_bytes(ser.read(2),'little') )

    # int_data=-511
    # ser.write(int_data.to_bytes(2,'little',signed=True))
    # print( int.from_bytes(ser.read(2),'little',signed=True) )

    # # 2バイト整数を送信　→　 2バイト受信　→　2バイト整数として解釈
    # # 負の整数の場合，to_bytes，from_bytes共にsigned=Trueを指定する
    # # バイトオブジェクトへの相互変換にstructオブジェクトを使用する場合
    # int_data=511
    # ser.write(struct.pack("<H",int_data))
    # print(struct.unpack("<H",ser.read(2))[0])
    
    # int_data=-511
    # ser.write(struct.pack("<h",int_data))
    # print(struct.unpack("<h",ser.read(2))[0])


    # # 通信プロトコルを用いる例．ヘッダー(MP)＋データ(3バイト)の形
    # cmd_header=b'MP'
    # cmd_data =[100,110,120]
    
    # # バイトオブジェクトへの相互変換にto_bytes，from_bytesを使用する場合
    # # 送信データの統合
    # cmd=cmd_header+cmd_data[0].to_bytes(1,'little')+cmd_data[1].to_bytes(1,'little')+cmd_data[2].to_bytes(1,'little')
    # print(len(cmd))
    # # データ送信
    # ser.write(cmd)
    # # データ受信＋解釈
    # print(ser.read(2))
    # print( int.from_bytes(ser.read(1),'little') )
    # print( int.from_bytes(ser.read(1),'little') )
    # print( int.from_bytes(ser.read(1),'little') )


    # # バイトオブジェクトへの相互変換にstructオブジェクトを使用する場合
    # # 送信データの統合
    # cmd_struct=cmd_header+struct.pack("<BBB",cmd_data[0],cmd_data[1],cmd_data[2])
    # print(len(cmd))
    #  # データ送信
    # ser.write(cmd)
    # # データ受信＋解釈
    # print(ser.read(2))
    # m0,m1,m2=struct.unpack("<BBB",ser.read(3))
    # print(m0,m1,m2)
    
    # #ポートクローズ
    # ser.close()

