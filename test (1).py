import AIO
import sys
from time import sleep
from collections import deque
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    #使用するCh数
    num_of_channels=8
    #グラフ表示したいCh番号
    visible_ch=0
    #計測周波数[Hz]
    sampling_rate=1000.0
    #信号を保存するバッファ長
    buffer_size=128
    #計測を打ち切るデータ数
    limit_of_sampling=10000
    #デバイス名・IDを検索
    device_name=AIO.queryAioDeviceName()

    try:
        #デバイス名を直接指定する場合
        aio=AIO.AIO(b'AIO000')

        #queryAioDeviceNameで利用できるデバイス名を探しそれを使用する場合
        #aio=AIO.AIO(device_name[0][0])

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
    aio.resetAiMemory()
    aio.startAi()
    print("Sampling start")


    #信号を保存するバッファを確保
    #使用するChの数だけ，長さbuffer_sizeのキュー(待ち行列FIFO)を用意
    signal_buffer = []
    for ch in range(0,num_of_channels):
        signal_buffer.append( deque([], maxlen=buffer_size) ) 
        signal_buffer[ch].extend([32767]*buffer_size)
    
    
    #######　グラフ表示用変数の初期化 #######　
    
    #長さbuffer_sizeのX軸用のデータをキューで用意
    time_stamp=deque([], maxlen=buffer_size)
    time_stamp.extend([0]*buffer_size)
    #インタラクティブモードをOnに
    plt.ion()
    #新しいウィンドウの描画
    plt.figure()
    ##初期化のため一度plotし，グラフのオブジェクトliを受け取る
    li, = plt.plot(time_stamp, signal_buffer[visible_ch])
    #軸の設定
    plt.title("Contec AD device: Ch.%d(Analog Input)"%visible_ch)
    plt.ylim(0, 65535)
    plt.xlabel("Step")
    plt.ylabel("Coverted data(ASCII)")

    ###########################################

    count=0

#   メインループ
    while True:

        #メモリ内のサンプリング可能なデータ数を取得
        num_of_sampling=aio.getAiSamplingCount()
        
        try:
            if num_of_sampling > buffer_size :
                num_of_sampling=buffer_size

            #メモリ内からnum_of_samplingだけデータを抽出
            #返り値はタプルで返ってくる．(int,list)
            #1項目intには実際にメモリから抽出できたデータ数が格納されている
            #　下記の例では，num_of_sampling[0]
            #2項目listには，メモリから抽出したデータがリストに格納されている[Ch数×データ数]
            #　下記の例にはないが，num_of_sampling[1][0]～num_of_sampling[1][(Ch数×データ数)-1]
 
            get_data=aio.getAiSamplingData(num_of_sampling)
        
        except ValueError as ve:
            
            print(ve)
        
        else:
            #取得した分のデータget_data[1]を各Chのバッファー(キュー)に追加
            # (追加した分だけ古いデータから削除される)
            #リストのスライス操作を利用しCh毎にデータを分割
            for ch in range(0,num_of_channels):
                signal_buffer[ch].extend(get_data[1][ch::num_of_channels])
            
            #######　グラフ表示　#######　
            #取得したデータに対応するStep(X軸)をリストで生成し，キューに追加
            time_stamp_cur=range(count,(count+get_data[0]))
            time_stamp.extend(time_stamp_cur)
            
            #グラフデータをセット
            li.set_xdata(time_stamp)    #X軸
            li.set_ydata(signal_buffer[visible_ch]) #Y軸
            #X軸の範囲を設定
            plt.xlim(min(time_stamp), max(time_stamp))
            #グラフ描画
            plt.draw()
            #描画するまでの時間
            plt.pause(.01)
            
            ###########################

            #サンプルできたデータ数を出力
            print(get_data[0])
            
            count+=get_data[0]

        
        #limit_of_samplin以上取得したら変換終了
        if count > limit_of_sampling:
            aio.stopAi()
            print("Sampling stop")
            break
