################################################################################################
# Contec API-USBP Libraly for python 3
#
#  * This library is revised version of pyAPIUSBP(http://pyapiusbp.sourceforge.net/) 
#    revised by Ryu Kato.
#   
################################################################################################
import ctypes
import warnings

try:
    DLL = ctypes.windll.LoadLibrary('caio.dll')
except:
    warnings.warn('caio.dll is not found. AIO module will not work.')
    DLL = None

#----------------------------------------------------------------------------------------------
# External Signal
#----------------------------------------------------------------------------------------------
AIO_AIF_CLOCK				 = 0	#AI external clock
AIO_AIF_START				 = 1	#AI external start trigger
AIO_AIF_STOP				 = 2	#AI external stop trigger
AIO_AOF_CLOCK				 = 3	#AO external clock
AIO_AOF_START				 = 4	#AO external start trigger
AIO_AOF_STOP				 = 5	#AO external stop trigger
AIO_ALLF					 = -1	#ALL

#----------------------------------------------------------------------------------------------
# Analog IO range
#----------------------------------------------------------------------------------------------
PM10						 = 0	#-10 to 10V
PM5							 = 1	#-5 to 5V
PM25						 = 2	#-2.5 to 2.5V
PM125						 = 3	#-1.25 to 1.25V
PM1							 = 4	#-1 to 1V
PM0625						 = 5	#-0.625 to 0.625V
PM05						 = 6	#-0.5 to 0.5V
PM03125						 = 7	#-0.3125 to 0.3125V
PM025						 = 8	#-0.25 to 0.25V
PM0125						 = 9	#-0.125 to 0.125V
PM01						 = 10	#-0.1 to 0.1V
PM005						 = 11	#-0.05 to 0.05V
PM0025						 = 12	#-0.025 to 0.025V
PM00125						 = 13	#-0.0125 to 0.0125V
PM001						 = 14	#-0.01 to 0.01V
P10							 = 50	#0 to 10V
P5							 = 51	#0 to 5V
P4095						 = 52	#0 to 4.095V
P25							 = 53	#0 to 2.5V
P125						 = 54	#0 to 1.25V
P1							 = 55	#0 to 1V
P05							 = 56	#0 to 0.5V
P025						 = 57	#0 to 0.25V
P01							 = 58	#0 to 0.1V
P005						 = 59	#0 to 0.05V
P0025						 = 60	#0 to 0.025V
P00125						 = 61	#0 to 0.0125V
P001						 = 62	#0 to 0.01V
P20MA						 = 100	#0 to 20mA
P4TO20MA					 = 101	#4 to 20mA
P1TO5						 = 150	#1 to 5V

#----------------------------------------------------------------------------------------------
# AI event
#----------------------------------------------------------------------------------------------
AIE_START			 = 0x00000002
AIE_RPTEND			 = 0x00000010
AIE_END				 = 0x00000020
AIE_DATA_NUM		 = 0x00000080
AIE_DATA_TSF		 = 0x00000100
AIE_OFERR			 = 0x00010000
AIE_SCERR			 = 0x00020000
AIE_ADERR			 = 0x00040000

#----------------------------------------------------------------------------------------------
# AO event
#----------------------------------------------------------------------------------------------
AOE_START			 = 0x00000002
AOE_RPTEND			 = 0x00000010
AOE_END				 = 0x00000020
AOE_DATA_NUM		 = 0x00000080
AOE_DATA_TSF		 = 0x00000100
AOE_SCERR			 = 0x00020000
AOE_DAERR			 = 0x00040000

#----------------------------------------------------------------------------------------------
# Counter event
#----------------------------------------------------------------------------------------------
CNTE_DATA_NUM		 = 0x00000010
CNTE_ORERR			 = 0x00010000
CNTE_ERR			 = 0x00020000

#----------------------------------------------------------------------------------------------
# Timer event
#----------------------------------------------------------------------------------------------
TME_INT				 = 0x00000001

#----------------------------------------------------------------------------------------------
# AI status
#----------------------------------------------------------------------------------------------
AIS_BUSY			 = 0x00000001
AIS_START_TRG		 = 0x00000002
AIS_DATA_NUM		 = 0x00000010
AIS_OFERR			 = 0x00010000
AIS_SCERR			 = 0x00020000
AIS_AIERR			 = 0x00040000
AIS_DRVERR			 = 0x00080000

#----------------------------------------------------------------------------------------------
# AO status
#----------------------------------------------------------------------------------------------
AOS_BUSY			 = 0x00000001
AOS_START_TRG		 = 0x00000002
AOS_DATA_NUM		 = 0x00000010
AOS_SCERR			 = 0x00020000
AOS_AOERR			 = 0x00040000
AOS_DRVERR			 = 0x00080000

#----------------------------------------------------------------------------------------------
# Counter status
#----------------------------------------------------------------------------------------------
CNTS_BUSY			 = 0x00000001
CNTS_DATA_NUM		 = 0x00000010
CNTS_ORERR			 = 0x00010000
CNTS_ERR			 = 0x00020000

#----------------------------------------------------------------------------------------------
# AI message
#----------------------------------------------------------------------------------------------
AIOM_AIE_START			 = 0x1000
AIOM_AIE_RPTEND			 = 0x1001
AIOM_AIE_END			 = 0x1002
AIOM_AIE_DATA_NUM		 = 0x1003
AIOM_AIE_DATA_TSF		 = 0x1007
AIOM_AIE_OFERR			 = 0x1004
AIOM_AIE_SCERR			 = 0x1005
AIOM_AIE_ADERR			 = 0x1006

#----------------------------------------------------------------------------------------------
# AO message
#----------------------------------------------------------------------------------------------
AIOM_AOE_START			 = 0x1020
AIOM_AOE_RPTEND			 = 0x1021
AIOM_AOE_END			 = 0x1022
AIOM_AOE_DATA_NUM		 = 0x1023
AIOM_AOE_DATA_TSF		 = 0x1027
AIOM_AOE_SCERR			 = 0x1025
AIOM_AOE_DAERR			 = 0x1026

#----------------------------------------------------------------------------------------------
# Counter message
#----------------------------------------------------------------------------------------------
AIOM_CNTE_DATA_NUM		 = 0x1042
AIOM_CNTE_ORERR			 = 0x1043
AIOM_CNTE_ERR			 = 0x1044

#----------------------------------------------------------------------------------------------
# Timer message
#----------------------------------------------------------------------------------------------
AIOM_TME_INT			 = 0x1060

#----------------------------------------------------------------------------------------------
# AI additional data
#----------------------------------------------------------------------------------------------
AIAT_AI				 = 0x00000001
AIAT_AO0			 = 0x00000100
AIAT_DIO0			 = 0x00010000
AIAT_CNT0			 = 0x01000000
AIAT_CNT1			 = 0x02000000

#----------------------------------------------------------------------------------------------
# Conter mode
#----------------------------------------------------------------------------------------------
CNT_LOADPRESET		 = 0x0000001
CNT_LOADCOMP		 = 0x0000002

#----------------------------------------------------------------------------------------------
# Event controller signal
#----------------------------------------------------------------------------------------------
AIOECU_DEST_AI_CLK			 = 4
AIOECU_DEST_AI_START		 = 0
AIOECU_DEST_AI_STOP			 = 2
AIOECU_DEST_AO_CLK			 = 36
AIOECU_DEST_AO_START		 = 32
AIOECU_DEST_AO_STOP			 = 34
AIOECU_DEST_CNT0_UPCLK		 = 134
AIOECU_DEST_CNT1_UPCLK		 = 135
AIOECU_DEST_CNT0_START		 = 128
AIOECU_DEST_CNT1_START		 = 129
AIOECU_DEST_CNT0_STOP		 = 130
AIOECU_DEST_CNT1_STOP		 = 131
AIOECU_DEST_MASTER1			 = 104
AIOECU_DEST_MASTER2			 = 105
AIOECU_DEST_MASTER3			 = 106

#----------------------------------------------------------------------------------------------
# Event controller origin
#----------------------------------------------------------------------------------------------
AIOECU_SRC_OPEN				 = -1
AIOECU_SRC_AI_CLK			 = 4
AIOECU_SRC_AI_EXTCLK		 = 146
AIOECU_SRC_AI_TRGSTART		 = 144
AIOECU_SRC_AI_LVSTART		 = 28
AIOECU_SRC_AI_STOP			 = 17
AIOECU_SRC_AI_STOP_DELAY	 = 18
AIOECU_SRC_AI_LVSTOP		 = 29
AIOECU_SRC_AI_TRGSTOP		 = 145
AIOECU_SRC_AO_CLK			 = 66
AIOECU_SRC_AO_EXTCLK		 = 149
AIOECU_SRC_AO_TRGSTART		 = 147
AIOECU_SRC_AO_STOP_FIFO		 = 352
AIOECU_SRC_AO_STOP_RING		 = 80
AIOECU_SRC_AO_TRGSTOP		 = 148
AIOECU_SRC_CNT0_UPCLK		 = 150
AIOECU_SRC_CNT1_UPCLK		 = 152
AIOECU_SRC_CNT0_CMP			 = 288
AIOECU_SRC_CNT1_CMP			 = 289
AIOECU_SRC_SLAVE1			 = 136
AIOECU_SRC_SLAVE2			 = 137
AIOECU_SRC_SLAVE3			 = 138
AIOECU_SRC_START			 = 384
AIOECU_SRC_STOP				 = 385

#----------------------------------------------------------------------------------------------
# Counter message for M-device 
#----------------------------------------------------------------------------------------------
AIOM_CNTM_COUNTUP_CH0		 = 0x1070
AIOM_CNTM_COUNTUP_CH1		 = 0x1071
AIOM_CNTM_TIME_UP			 = 0x1090
AIOM_CNTM_COUNTER_ERROR		 = 0x1091
AIOM_CNTM_CARRY_BORROW		 = 0x1092

CONST_FIFO = 0
CONST_RING = 1
CONST_DEVICE_BUFFER = 0
CONST_USER_BUFFER = 1
CONST_SINGLE_END = 0
CONST_DIFFERENTIAL = 1
CONST_MEMORY_NO_OVERWRITE = 0
CONST_MEMORY_OVERWRITE = 1
CONST_MEMORY_NO_REPEAT = 0
CONST_MEMORY_REPEAT = 1
CONST_FALLING_EDGE = 0
CONST_RISING_EDGE = 1
CONST_BOTH_DIRECTION = 0
CONST_RISING_DIRECTION = 1
CONST_FALLING_DIRECTION = 2
CONST_AIO_EXTERNAL_SIGNALS = [AIO_AIF_CLOCK, AIO_AIF_START, AIO_AIF_STOP, AIO_AOF_CLOCK, AIO_AOF_START, AIO_AOF_STOP, AIO_ALLF]

def queryAioDeviceName():
    """
    Get a list of AIO devices. Information of each device consists of device ID
    and device name (example: ['AIO000', 'AIO-120802LN-USB']).
    
    :return:
        A list of AIO device names.
    """
    devlist = []
    deviceName = (ctypes.c_char*256)()
    device = (ctypes.c_char*256)()
    for i in range(255):
        ret = DLL.AioQueryDeviceName(i, ctypes.byref(deviceName), ctypes.byref(device))
        if ret == 0: #success
            devlist.append([deviceName.value, device.value])
            break
    return devlist

def getAioDeviceType(device):
    """
    Get type of the device. Device name can be obtained as the second element 
    of device information returned by :func:`~pyAPISUB.AIO.getAioDeviceType`.
    
    :param str device:
        Model name of a device.
    :return:
        Device type.
    """
    deviceType = ctypes.c_short()
    ret = DLL.AioGetDeviceType(device, ctypes.byref(deviceType))
    if ret != 0: #failed
        msg = (ctypes.c_char*256)()
        DLL.AioGetErrorString(ret,ctypes.byref(msg))
        raise ValueError('AioGetDeviceType failed (%s)' % msg.value)
    if deviceType.value == 1:
        return 'PCI'
    elif deviceType.value == 2:
        return 'PCICard'
    elif deviceType.value == 3:
        return 'USB'
    elif deviceType.value == 5:
        return 'DemoDevice'
    elif deviceType.value == 6:
        return 'CardBus'

def getRangeString(r):
    """
    Get range string from range ID.
    
    :param int r:
        range ID.
    :return:
        range string.
    """
    if r==PM10:
        return 'PM10'
    if r==PM5:
        return 'PM5'
    if r==PM25:
        return 'PM25'
    if r==PM125:
        return 'PM125'
    if r==PM1:
        return 'PM1'
    if r==PM0625:
        return 'PM0625'
    if r==PM05:
        return 'PM05'
    if r==PM03125:
        return 'PM03125'
    if r==PM025:
        return 'PM025'
    if r==PM0125:
        return 'PM0125'
    if r==PM01:
        return 'PM01'
    if r==PM005:
        return 'PM005'
    if r==PM0025:
        return 'PM0025'
    if r==PM00125:
        return 'PM00125'
    if r==PM001:
        return 'PM001'
    if r==P10:
        return 'P10'
    if r==P5:
        return 'P5'
    if r==P4095:
        return 'P4095'
    if r==P25:
        return 'P25'
    if r==P125:
        return 'P125'
    if r==P1:
        return 'P1'
    if r==P05:
        return 'P05'
    if r==P025:
        return 'P025'
    if r==P01:
        return 'P01'
    if r==P005:
        return 'P005'
    if r==P0025:
        return 'P0025'
    if r==P00125:
        return 'P00125'
    if r==P001:
        return 'P001'
    if r==P20MA:
        return 'P20MA'
    if r==P4TO20MA:
        return 'P4T020MA'
    if r==P1TO5:
        return 'P1T05'
    
class AIO(object):
    def __init__(self, deviceName):
        """
        Constructor of AIO object.
        
        :param str deviceName:
            Name of device name. Use :func:`~pyAPIUSBP.Aio.queryAIODeviceName`
            to query device name.
        """
        cId = ctypes.c_short()
        ret = DLL.AioInit(deviceName, ctypes.byref(cId))
        if ret != 0: #failed
            raise ValueError( 'AioInit failed (%s)' % self.getErrorString(ret).decode('sjis'))
        self.Id = cId.value
        
    def __del__(self):
        """
        Destructor of AIO object.
        """
        ret = DLL.AioExit(self.Id)
        if ret != 0: #failed
            raise ValueError('AioExit failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    #------------------------------------------------------------------------------------------
    # Common functions
    #------------------------------------------------------------------------------------------
    def getErrorString(self, ret):
        """
        This method translates error code to error string.
        
        :param int code:
            Error code of Contec API-USBP AIO functions.
        """
        msg = (ctypes.c_char*256)()
        DLL.AioGetErrorString(ret,ctypes.byref(msg))
        return msg.value
    
    def resetProcess(self):
        """
        See document of AioResetProcess of API-USBP.
        """
        ret = DLL.AioResetProcess(self.Id)
        if ret != 0: #failed
            raise ValueError( 'AioResetProcess failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def resetDevice(self):
        """
        This method resets device. Exception is raised if reset fails.
        """
        ret = DLL.AioResetDevice(self.Id)
        if ret != 0: #failed
            raise ValueError('AioResetDevice failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def setControlFilter(self, signal, value):
        """
        This method sets digital filter for external signal.
        See document of AioSetControlFilter of API-USBP.
        
        :param int signal:
            See document of AioSetControlFilter
        :param value:
            See document of AioSetControlFilter
        """
        if not signal in CONST_AIO_EXTERNAL_SIGNALS:
            raise ValueError('Invalid signal')
        if not value in [0, 0.05, 1, 10, 100, 128, 16000]:
            raise ValueError('Invalid value')
        
        ret = DLL.AioSetControlFilter(self.Id, signal, value)
        if ret != 0: #failed
            raise ValueError('AioSetControlFilter failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getControlFilter(self, signal):
        """
        This method gets current status of digital filter for external signal.
        See document of AioGetControlFilter of API-USBP.
        
        :param int signal:
            See document of AioGetControlFilter
        :return:
            See document of AioGetControlFilter
        """
        if not signal in CONST_AIO_EXTERNAL_SIGNALS:
            raise ValueError('Invalid signal')
        value = ctypes.c_float()
        ret = DLL.AioGetControlFilter(self.Id, signal, ctypes.byref(value))
        if ret != 0: #failed
            raise ValueError('AioGetControlFilter failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return value.value
    
    #------------------------------------------------------------------------------------------
    # Analog Input
    #------------------------------------------------------------------------------------------
    #----- Simple methods -----
    def singleAi(self, channel):
        """
        This method reads an analog input channel.
        
        :param int channel:
            Channel number.
        :return:
            Value (range depends on the device).
        """
        data = ctypes.c_ulong()
        ret = DLL.AioSingleAi(self.Id, channel, ctypes.byref(data))
        if ret != 0: #failed
            raise ValueError('AioSingleAi failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return data.value
    
    def singleAiEx(self, channel):
        """
        This method reads an analog input channel and converts it to
        voltage/current.
        
        :param int channel:
            Channel number.
        :return:
            Value (range depends on the device).
        """
        data = ctypes.c_float()
        ret = DLL.AioSingleAiEx(self.Id, channel, ctypes.byref(data))
        if ret != 0: #failed
            raise ValueError('AioSingleAi failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return data.value
    
    def multiAi(self, channelList):
        """
        This method reads multiple analog input channels.
        
        :param list channelList:
            List of channel numbers.
        :return:
            List of values (range of values depends on the device).
        """
        nch = len(channelList)
        data = (ctypes.c_long*nch)()
        ret = DLL.AioMultiAi(self.Id, nch, ctypes.byref(data))
        if ret != 0: #failed
            raise ValueError('AioMultiAi failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return list(data)
    
    def multiAiEx(self, channelList):
        """
        This method reads multiple analog input channels and converts them to
        voltage/current.
        
        :param list channelList:
            List of channel numbers.
        :return:
            List of values (range of values depends on the device).
        """
        nch = len(channelList)
        data = (ctypes.c_long*nch)()
        ret = DLL.AioMultiAiEx(self.Id, nch, ctypes.byref(data))
        if ret != 0: #failed
            raise ValueError('AioMultiAiEx failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return list(data)
    
    #----- Resolution -----
    def getAiResolution(self):
        """
        This function returns resolution of analog input.
        
        :return:
            0 for no-analog-input, 12 for 12bit resolution,
            16 for 16bit resolution.
        """
        resolution = ctypes.c_short()
        ret = DLL.AioGetAiResolution(self.Id, ctypes.byref(resolution))
        if ret != 0: #failed
            raise ValueError('AioGetAiResolution failed (%s)' % self.getErrorString(ret).decode('sjis'))
        
    #----- Input method -----
    def setAiInputMethod(self, method):
        """
        This function sets input method of analog inputs.
        
        :param int method:
            0 for single-ended, 1 for differential.
        """
        if not method in [CONST_SINGLE_END, CONST_DIFFERENTIAL]:
            raise ValueError('method must be CONST_SINGLE_END or CONST_DIFFERENTIAL.')
        ret = DLL.AioSetAiInputMethod(self.Id, method)
        if ret != 0: #failed
            raise ValueError('AioSetAiInputMethod failed (%s)' % self.getErrorString(ret).decode('sjis'))
        
    def getAiInputMethod(self):
        """
        This function returns current input method of analog inputs.
        
        :return:
            0 for single-ended, 1 for differential.
        """
        method = ctypes.c_short()
        ret = DLL.AioGetAiInputMethod(self.Id, ctypes.byref(method))
        if ret != 0: #failed
            raise ValueError('AioGetAiInputMethod failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return method.value
    
    #----- Channel settings -----
    def getAiMaxChannels(self):
        """
        This function returns maximum number of analog input channels.
        
        :return:
            Maximum number of analog input channels.
        """
        channels = ctypes.c_short()
        ret = DLL.AioGetAiMaxChannels(self.Id, ctypes.byref(channels))
        if ret != 0: #failed
            raise ValueError('AioSetAiMaxChannels failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return channels.value
        
    def setAiChannels(self, channels):
        """
        This function sets number of channels for analog input.
        
        :param int channels:
            Number of channels.
        """
        ret = DLL.AioSetAiChannels(self.Id, channels)
        if ret != 0: #failed
            raise ValueError('AioSetAiChannels failed (%s)' % self.getErrorString(ret).decode('sjis'))
        
    def getAiChannels(self):
        """
        This function returns number of channels currently used for analog
        input.
        
        :return:
            Number of channels.
        """
        channels = ctypes.c_short()
        ret = DLL.AioGetAiChannels(self.Id, ctypes.byref(channels))
        if ret != 0: #failed
            raise ValueError('AioSetAiChannels failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return channels.value
    
    def setAiChannelSequence(self, sequence, channel):
        """
        This function sets order of AD conversion across channels.
        See document of AioSetAiChannelSequence of API-USBP.
        
        :param list sequence:
            See document of AioSetAiChannelSequence of API-USBP.
        :param int channel:
            See document of AioSetAiChannelSequence of API-USBP.
        """
        ret = DLL.AioSetAiChannelSequence(self.Id, sequence, channel)
        if ret != 0: #failed
            raise ValueError('AioSetAiChannelSequence failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAiChannelSequence(self, sequence):
        """
        This function gets current order of AD conversion across channels.
        See document of AioGetAiChannelSequence of API-USBP.
        
        :param list sequence:
            See document of AioGetAiChannelSequence of API-USBP.
        :return:
            See document of AioGwetAiChannelSequence of API-USBP.
        """
        channel = ctypes.c_short()
        ret = DLL.AioGetAiChannelSequence(self.Id, sequence, ctypes.byref(channel))
        if ret != 0: #failed
            raise ValueError('AioGetAiChannelSequence failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return channel.value
    
    #----- Range settings -----
    def setAiRange(self, channel, AiRange):
        """
        This method sets range of an analog input channel.
        
        :param int channel:
            Analog input channel.
        :param int AiRange:
            Range ID. :class:`pyAPISUBP.AIO` has Range ID as class attributes.
            That is, you can specify range like pyAPISUBP.AIO.P5
        """
        ret = DLL.AioSetAiRange(self.Id, channel, AiRange)
        if ret != 0: #failed
            raise ValueError('AioSetAiRange failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def setAiRangeAll(self, AiRange):
        """
        This method sets range of all analog input channels.
        
        :param int AiRange:
            Range ID. :class:`pyAPISUBP.AIO` has Range ID as class attributes.
            That is, you can specify range like pyAPISUBP.AIO.P5
        """
        ret = DLL.AioSetAiRangeAll(self.Id, AiRange)
        if ret != 0: #failed
            raise ValueError('AioSetAiRangeAll failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAiRange(self, channel):
        """
        This method returns range an analog input channel.
        
        :param int channel:
            Analog input channel.
        :return:
            Range ID. :func:`~pyAPISUBP.AIO.getRangeString` can be used to get 
            range string from range IO.
        """
        AiRange = ctypes.c_short()
        ret = DLL.AioGetAiChannels(self.Id, ctypes.byref(AiRange))
        if ret != 0: #failed
            raise ValueError('AioSetAiRange failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return AiRange.value
    
    #----- Transfer modes -----
    def setAiTransferMode(self, mode):
        """
        This method sets data transfer method of analog input.
        
        :param int mode:
            pyAPIUSBP.AIO.CONST_DEVICE_BUFFER for device-buffer-mode,
            pyAPIUSBP.AIO.CONST_USER_BUFFER for user-buffer-mode.
        """
        if not mode in [CONST_DEVICE_BUFFER, CONST_USER_BUFFER]:
            raise ValueError('mode must be CONST_DEVICE_BUFFER or CONST_USER_BUFFER.')
        ret = DLL.AioSetAiTransferMode(self.Id, mode)
        if ret != 0: #failed
            raise ValueError('AioSetAiTransferMode failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAiTransferMode(self):
        """
        This method returns current data transfer method of analog input.
        
        :return:
            pyAPIUSBP.AIO.CONST_DEVICE_BUFFER for device-buffer-mode,
            pyAPIUSBP.AIO.CONST_USER_BUFFER for user-buffer-mode.
        """
        mode = ctypes.c_short()
        ret = DLL.AioGetAiTransferMode(self.Id, ctypes.byref(mode))
        if ret != 0: #failed
            raise ValueError('AioGetAiTransferMode failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return mode.value
    
    #----- Memory types -----
    def setAiMemoryType(self, memType):
        """
        This method sets memory type for data recording.
        See document of AioSetAiMemoryType of API-USBP.
        
        :param int memType:
            See document of AioSetAiMemoryType of API-USBP.
        """
        if not memType in [CONST_FIFO, CONST_RING, CONST_MEMORY_NO_OVERWRITE, CONST_MEMORY_OVERWRITE]:
            raise ValueError('memType must be CONST_FIFO, CONST_RING, CONST_MEMORY_NO_OVERWRITE or CONST_MEMORY_OVERWRITE')
        ret = DLL.AioSetAiMemoryType(self.Id, memType)
        if ret != 0: #failed
            raise ValueError('AioSetAiMemoryType failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAiMemoryType(self):
        """
        This method returns current memory type for data recording.
        See document of AioGetAiMemoryType of API-USBP.
        
        :return:
            See document of AioGetAiMemoryType of API-USBP.
        """
        memType = ctypes.c_short()
        ret = DLL.AioGetAiMemoryType(self.Id, ctypes.byref(memType))
        if ret != 0: #failed
            raise ValueError('AioGetAiMemoryType failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return memType.value
    
    #----- Clock settings -----
    def setAiClockType(self, clockType):
        """
        See document of AioSetAiClockType of API-USBP.
        """
        ret = DLL.AioSetAiClockType(self.Id, clockType)
        if ret != 0: #failed
            raise ValueError('AioSetAiClockType failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAiClockType(self):
        """
        See document of AioGetAiClockType of API-USBP.
        """
        clockType = ctypes.c_short()
        ret = DLL.AioGetAiClockType(self.Id, ctypes.byref(clockType))
        if ret != 0: #failed
            raise ValueError('AioGetAiClockType failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return clockType.value
    
    def setAiScanClock(self, clock):
        """
        See document of AioSetAiScanClock of API-USBP.
        """
        ret = DLL.AioSetAiScanClock(self.Id, clock)
        if ret != 0: #failed
            raise ValueError('AioSetAiScanClock failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAiScanClock(self):
        """
        See document of AioGetAiScanClock of API-USBP.
        """
        clock = ctypes.c_float()
        ret = DLL.AioGetAiScanClock(self.Id, ctypes.byref(clock))
        if ret != 0: #failed
            raise ValueError('AioGetAiScanClock failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return clock.value
    
    def setAiSamplingClock(self, clock):
        """
        See document of AioSetAiSamplingClock of API-USBP.
        """
        #cclock=ctypes.c_float(clock)
        ret = DLL.AioSetAiSamplingClock(self.Id, ctypes.c_float(clock))
        if ret != 0: #failed
            raise ValueError('AioSetAiSamplingClock failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAiSamplingClock(self):
        """
        See document of AioGetAiSamplingClock of API-USBP.
        """
        clock = ctypes.c_float()
        ret = DLL.AioGetAiSamplingClock(self.Id, ctypes.byref(clock))
        if ret != 0: #failed
            raise ValueError('AioGetAiSamplingClock failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return clock.value
    
    def setAiClockEdge(self, edge):
        """
        See document of AioSetAiClockEdge of API-USBP.
        """
        if not edge in [CONST_FALLING_EDGE, CONST_RISING_EDGE]:
            raise ValueError('Edge must be CONST_FALLING_EDGE or CONST_RISING_EDGE.')
        ret = DLL.AioSetAiClockEdge(self.Id, edge)
        if ret != 0: #failed
            raise ValueError('AioSetAiClockEdge failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAiClockEdge(self):
        """
        See document of AioGetAiClockEdge of API-USBP.
        """
        edge = ctypes.c_short()
        ret = DLL.AioGetAiClockEdge(self.Id, ctypes.byref(edge))
        if ret != 0: #failed
            raise ValueError('AioGetAiClockEdge failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return edge.value
    
    #----- Start trigger -----
    def setAiStartTrigger(self, trigger):
        """
        See document of AioSetAiStartTrigger of API-USBP.
        """
        ret = DLL.AioSetAiStartTrigger(self.Id, trigger)
        if ret != 0: #failed
            raise ValueError('AioSetAiStartTrigger failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAiStartTrigger(self):
        """
        See document of AioGetAiStartTrigger of API-USBP.
        """
        trigger = ctypes.c_short()
        ret = DLL.AioGetAiStartTrigger(self.Id, ctypes.byref(trigger))
        if ret != 0: #failed
            raise ValueError('AioGetAiStartTrigger failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return trigger.value
    
    def setAiStartLevel(self, channel, level, direction):
        """
        See document of AioSetAiStartLevel of API-USBP.
        """
        if not direction in [CONST_BOTH_DIRECTION, CONST_RISING_DIRECTION, CONST_FALLING_DIRECTION]:
            raise ValueError('direction must be CONST_BOTH_DIRECTION, CONST_RISING_DIRECTION or CONST_FALLING_DIRECTION.')
        ret = DLL.AioSetAiStartLevel(self.Id, channel, level, direction)
        if ret != 0: #failed
            raise ValueError('AioSetAiStartLevel failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def setAiStartLevelEx(self, channel, level, direction):
        """
        See document of AioSetAiStartLevelEx of API-USBP.
        """
        if not direction in [CONST_BOTH_DIRECTION, CONST_RISING_DIRECTION, CONST_FALLING_DIRECTION]:
            raise ValueError('direction must be CONST_BOTH_DIRECTION, CONST_RISING_DIRECTION or CONST_FALLING_DIRECTION.')
        ret = DLL.AioSetAiStartLevelEx(self.Id, channel, level, direction)
        if ret != 0: #failed
            raise ValueError('AioSetAiStartLevelEx failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAiStartLevel(self, channel):
        """
        See document of AioGetAiStartLevel of API-USBP.
        """
        level = ctypes.c_long()
        direction = ctypes.c_short()
        ret = DLL.AioGetAiStartLevel(self.Id, channel, ctypes.byref(level), ctypes.byref(direction))
        if ret != 0: #failed
            raise ValueError('AioGetAiStartLevel failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return (level.value, direction.value)
    
    def getAiStartLevelEx(self, channel):
        """
        See document of AioGetAiStartLevelEx of API-USBP.
        """
        level = ctypes.c_float()
        direction = ctypes.c_short()
        ret = DLL.AioGetAiStartLevelEx(self.Id, channel, ctypes.byref(level), ctypes.byref(direction))
        if ret != 0: #failed
            raise ValueError('AioGetAiStartLevelEx failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return (level.value, direction.value)
    
    def setAiStartInRange(self, channel, level1, level2, stateTimes):
        """
        See document of AioSetAiStartInRange of API-USBP.
        """
        ret = DLL.AioSetAiStartInRange(self.Id, channel, level1, level2, stateTimes)
        if ret != 0: #failed
            raise ValueError('AioSetAiStartInRange failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def setAiStartInRangeEx(self, channel, level1, level2, stateTimes):
        """
        See document of AioSetAiStartInRangeEx of API-USBP.
        """
        ret = DLL.AioSetAiStartInRangeEx(self.Id, channel, level1, level2, stateTimes)
        if ret != 0: #failed
            raise ValueError('AioSetAiStartInRange failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAiStartInRange(self, channel):
        """
        See document of AioGetAiStartInRange of API-USBP.
        """
        level1 = ctypes.c_long()
        level2 = ctypes.c_long()
        stateTimes = ctypes.c_short()
        ret = DLL.AioGetAiStartInRange(self.Id, channel, ctypes.byref(level1), ctypes.byref(level1), ctypes.byref(stateTimes))
        if ret != 0: #failed
            raise ValueError('AioGetAiStartInRange failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return (level1.value, level2.value, stateTimes.value)
    
    def getAiStartInRangeEx(self, channel):
        """
        See document of AioGetAiStartInRangeEx of API-USBP.
        """
        level1 = ctypes.c_float()
        level2 = ctypes.c_float()
        stateTimes = ctypes.c_short()
        ret = DLL.AioGetAiStartInRangeEx(self.Id, channel, ctypes.byref(level1), ctypes.byref(level2), ctypes.byref(stateTimes))
        if ret != 0: #failed
            raise ValueError('AioGetAiStartInRange failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return (level1.value, level2.value)
    
    def setAiStartOutRange(self, channel, level1, level2, stateTimes):
        """
        See document of AioSetAiStartOutRange of API-USBP.
        """
        ret = DLL.AioSetAiStartOutRange(self.Id, channel, level1, level2, stateTimes)
        if ret != 0: #failed
            raise ValueError('AioSetAiStartOutRange failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def setAiStartOutRangeEx(self, channel, level1, level2, stateTimes):
        """
        See document of AioSetAiStartOutRangeEx of API-USBP.
        """
        ret = DLL.AioSetAiStartOutRangeEx(self.Id, channel, level1, level2, stateTimes)
        if ret != 0: #failed
            raise ValueError('AioSetAiStartOutRange failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAiStartOutRange(self, channel):
        """
        See document of AioGetAiStartOutRange of API-USBP.
        """
        level1 = ctypes.c_long()
        level2 = ctypes.c_long()
        stateTimes = ctypes.c_short()
        ret = DLL.AioGetAiStartOutRange(self.Id, channel, ctypes.byref(level1), ctypes.byref(level1), ctypes.byref(stateTimes))
        if ret != 0: #failed
            raise ValueError('AioGetAiStartOutRange failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return (level1.value, level2.value, stateTimes.value)
    
    def getAiStartOutRangeEx(self, channel):
        """
        See document of AioGetAiStartOutRangeEx of API-USBP.
        """
        level1 = ctypes.c_float()
        level2 = ctypes.c_float()
        stateTimes = ctypes.c_short()
        ret = DLL.AioGetAiStartOutRangeEx(self.Id, channel, ctypes.byref(level1), ctypes.byref(level2), ctypes.byref(stateTimes))
        if ret != 0: #failed
            raise ValueError('AioGetAiStartOutRange failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return (level1.value, level2.value)
    
    #----- Stop trigger -----
    def setAiStopTrigger(self, trigger):
        """
        See document of AioSetAiStopTrigger of API-USBP.
        """
        ret = DLL.AioSetAiStopTrigger(self.Id, trigger)
        if ret != 0: #failed
            raise ValueError('AioSetAiStopTrigger failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAiStopTrigger(self):
        """
        See document of AioGetAiStopTrigger of API-USBP.
        """
        trigger = ctypes.c_short()
        ret = DLL.AioGetAiStopTrigger(self.Id, ctypes.byref(trigger))
        if ret != 0: #failed
            raise ValueError('AioGetAiStopTrigger failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return trigger.value
    
    def setAiStopTimes(self, stopTimes):
        """
        See document of AioSetAiStopTimes of API-USBP.
        """
        ret = DLL.AioSetAiStopTimes(self.Id, stopTimes)
        if ret != 0: #failed
            raise ValueError('AioSetAiStopTimes failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAiStopTimes(self):
        """
        See document of AioGetAiStopTimes of API-USBP.
        """
        stopTimes = ctypes.c_long()
        ret = DLL.AioGetAiStopTimes(self.Id, ctypes.byref(stopTimes))
        if ret != 0: #failed
            raise ValueError('AioGetAiStopTimes failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return stopTimes.value
    
    def setAiStopLevel(self, channel, level, direction):
        """
        See document of AioSetAiStopLevel of API-USBP.
        """
        if not direction in [CONST_BOTH_DIRECTION, CONST_RISING_DIRECTION, CONST_FALLING_DIRECTION]:
            raise ValueError('direction must be CONST_BOTH_DIRECTION, CONST_RISING_DIRECTION or CONST_FALLING_DIRECTION.')
        ret = DLL.AioSetAiStopLevel(self.Id, channel, level, direction)
        if ret != 0: #failed
            raise ValueError('AioSetAiStopLevel failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def setAiStopLevelEx(self, channel, level, direction):
        """
        See document of AioSetAiStopLevelEx of API-USBP.
        """
        if not direction in [CONST_BOTH_DIRECTION, CONST_RISING_DIRECTION, CONST_FALLING_DIRECTION]:
            raise ValueError('direction must be CONST_BOTH_DIRECTION, CONST_RISING_DIRECTION or CONST_FALLING_DIRECTION.')
        ret = DLL.AioSetAiStopLevelEx(self.Id, channel, level, direction)
        if ret != 0: #failed
            raise ValueError('AioSetAiStopLevelEx failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAiStopLevel(self, channel):
        """
        See document of AioGetAiStopLevel of API-USBP.
        """
        level = ctypes.c_long()
        direction = ctypes.c_short()
        ret = DLL.AioGetAiStopLevel(self.Id, channel, ctypes.byref(level), ctypes.byref(direction))
        if ret != 0: #failed
            raise ValueError('AioGetAiStopLevel failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return (level.value, direction.value)
    
    def getAiStopLevelEx(self, channel):
        """
        See document of AioGetAiStopLevelEx of API-USBP.
        """
        level = ctypes.c_float()
        direction = ctypes.c_short()
        ret = DLL.AioGetAiStopLevelEx(self.Id, channel, ctypes.byref(level), ctypes.byref(direction))
        if ret != 0: #failed
            raise ValueError('AioGetAiStopLevelEx failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return (level.value, direction.value)
    
    def setAiStopInRange(self, channel, level1, level2, stateTimes):
        """
        See document of AioSetAiStopInRange of API-USBP.
        """
        ret = DLL.AioSetAiStopInRange(self.Id, channel, level1, level2, stateTimes)
        if ret != 0: #failed
            raise ValueError('AioSetAiStopInRange failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def setAiStopInRangeEx(self, channel, level1, level2, stateTimes):
        """
        See document of AioSetAiStopInRangeEx of API-USBP.
        """
        ret = DLL.AioSetAiStopInRangeEx(self.Id, channel, level1, level2, stateTimes)
        if ret != 0: #failed
            raise ValueError('AioSetAiStopInRange failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAiStopInRange(self, channel):
        """
        See document of AioGetAiStopInRange of API-USBP.
        """
        level1 = ctypes.c_long()
        level2 = ctypes.c_long()
        stateTimes = ctypes.c_short()
        ret = DLL.AioGetAiStopInRange(self.Id, channel, ctypes.byref(level1), ctypes.byref(level1), ctypes.byref(stateTimes))
        if ret != 0: #failed
            raise ValueError('AioGetAiStopInRange failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return (level1.value, level2.value, stateTimes.value)
    
    def getAiStopInRangeEx(self, channel):
        """
        See document of AioGetAiStopInRangeEx of API-USBP.
        """
        level1 = ctypes.c_float()
        level2 = ctypes.c_float()
        stateTimes = ctypes.c_short()
        ret = DLL.AioGetAiStopInRangeEx(self.Id, channel, ctypes.byref(level1), ctypes.byref(level2), ctypes.byref(stateTimes))
        if ret != 0: #failed
            raise ValueError('AioGetAiStopInRange failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return (level1.value, level2.value)
    
    def setAiStopOutRange(self, channel, level1, level2, stateTimes):
        """
        See document of AioSetAiStopOutRange of API-USBP.
        """
        ret = DLL.AioSetAiStopOutRange(self.Id, channel, level1, level2, stateTimes)
        if ret != 0: #failed
            raise ValueError('AioSetAiStopOutRange failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def setAiStopOutRangeEx(self, channel, level1, level2, stateTimes):
        """
        See document of AioSetAiStopOutRangeEx of API-USBP.
        """
        ret = DLL.AioSetAiStopOutRangeEx(self.Id, channel, level1, level2, stateTimes)
        if ret != 0: #failed
            raise ValueError('AioSetAiStopOutRange failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAiStopOutRange(self, channel):
        """
        See document of AioGetAiStopOutRange of API-USBP.
        """
        level1 = ctypes.c_long()
        level2 = ctypes.c_long()
        stateTimes = ctypes.c_short()
        ret = DLL.AioGetAiStopOutRange(self.Id, channel, ctypes.byref(level1), ctypes.byref(level1), ctypes.byref(stateTimes))
        if ret != 0: #failed
            raise ValueError('AioGetAiStopOutRange failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return (level1.value, level2.value, stateTimes.value)
    
    def getAiStopOutRangeEx(self, channel):
        """
        See document of AioGetAiStopOutRangeEx of API-USBP.
        """
        level1 = ctypes.c_float()
        level2 = ctypes.c_float()
        stateTimes = ctypes.c_short()
        ret = DLL.AioGetAiStopOutRangeEx(self.Id, channel, ctypes.byref(level1), ctypes.byref(level2), ctypes.byref(stateTimes))
        if ret != 0: #failed
            raise ValueError('AioGetAiStopOutRange failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return (level1.value, level2.value)
    
    #----- Delay time -----
    def setAiStopDelayTimes(self, delayTimes):
        """
        See document of AioSetAiStopDelayTimes of API-USBP.
        """
        ret = DLL.AioSetAiStopDelayTimes(self.Id, delayTimes)
        if ret != 0: #failed
            raise ValueError('AioSetAiStopDelayTimes failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAiStopDelayTimes(self):
        """
        See document of AioSetAiStopDelayTimes of API-USBP.
        """
        delayTimes = ctypes.c_long()
        ret = DLL.AioSetAiStopDelayTimes(self.Id, ctypes.byref(delayTimes))
        if ret != 0: #failed
            raise ValueError('AioGetAiStopDelayTimes failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return delayTimes.value
    
    #----- Repeat times -----
    def setAiRepeatTimes(self, repeatTimes):
        """
        See document of AioSetAiRepeatTimes of API-USBP.
        """
        ret = DLL.AioSetAiRepeatTimes(self.Id, repeatTimes)
        if ret != 0: #failed
            raise ValueError('AioSetAiRepeatTimes failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAiRepeatTimes(self):
        """
        See document of AioGetAiRepeatTimes of API-USBP.
        """
        repeatTimes = ctypes.c_long()
        ret = DLL.AioGetAiRepeatTimes(self.Id, ctypes.byref(repeatTimes))
        if ret != 0: #failed
            raise ValueError('AioGetAiRepeatTimes failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return repeatTimes.value
    
    #----- Events -----
    def setAiEvent(self, hWnd, event):
        """
        This method sets events for analog input.
        See document of AioSetAiEvent of API-USBP.
        
        :param HWND hWnd:
            See document of AioSetAiEvent of API-USBP.
        :param int event:
            See document of AioSetAiEvent of API-USBP.
        """
        ret = DLL.AioSetAiEvent(self.Id, hWnd, event)
        if ret != 0: #failed
            raise ValueError('AioSetAiEvent failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAiEvent(self):
        """
        This method returns events for analog input.
        See document of AioGetAiEvent of API-USBP.
        
        :return:
            See document of AioGetAiEvent of API-USBP.
        """
        hWnd = ctypes.c_int()
        event = ctypes.c_long()
        ret = DLL.AioGetAiEvent(self.Id, ctypes.byref(hWnd), ctypes.byref(event))
        if ret != 0: #failed
            raise ValueError('AioGetAiEvent failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return (hWnd.value, event.value)
        
    def setAiCallBackProc(self, callbackProc, event, param):
        """
        This method sets callback function for event.
        See document of AioSetAiCallBackProc of API-USBP.
        
        :param ctypes.WINFUNCTYPE callbackProc:
            See document of AioSetAiCallBackProc of API-USBP.
        :param int event:
            See document of AioSetAiCallBackProc of API-USBP.
        :param int param:
            See document of AioSetAiCallBackProc of API-USBP.
        """
        ret = DLL.AioSetAiCallBackProc(self.Id, callbackProc, event, param)
        if ret != 0: #failed
            raise ValueError('AioSetAiCallBackProc Failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def setAiEventSamplingTimes(self, samplingTimes):
        """
        This method sets number of sampling times.
        See document of AioSetAiEventSamplingTimes of API-USBP.
        
        :param int samplingTimes:
            See document of AioSetAiEventSamplingTimes of API-USBP.
        """
        ret = DLL.AioSetAiEventSamplingTimes(self.Id , samplingTimes)
        if ret != 0: #failed
            raise ValueError('AioSetAiEventSamplingTimes Failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAiEventSamplingTimes(self):
        """
        This method returns current number of sampling times.
        See document of AioGetAiEventSamplingTimes of API-USBP.
        
        :return:
            See document of AioGetAiEventSamplingTimes of API-USBP.
        """
        samplingTimes = ctypes.c_long()
        ret = DLL.AioGetAiEventSamplingTimes(self.Id , ctypes.byref(samplingTimes))
        if ret != 0: #failed
            raise ValueError('AioGetAiEventSamplingTimes Failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def setAiEventTransferTimes(self, transferTimes):
        """
        This method sets number of transfer times.
        See document of AioSetAiEventTransferTimes of API-USBP.
        
        :param int transferTimes:
            See document of AioSetAiEventTransferTimes of API-USBP.
        """
        ret = DLL.AioSetAiEventTransferTimes(self.Id , transferTimes)
        if ret != 0: #failed
            raise ValueError('AioSetAiEventTransferTimes Failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAiEventTransferTimes(self):
        """
        This method returns current number of transfer times.
        See document of AioGetAiEventTransferTimes of API-USBP.
        
        :return:
            See document of AioGetAiEventTransferTimes of API-USBP.
        """
        transferTimes = ctypes.c_long()
        ret = DLL.AioGetAiEventTransferTimes(self.Id , ctypes.byref(transferTimes))
        if ret != 0: #failed
            raise ValueError('AioGetAiEventTransferTimes Failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    #----- Start/Stop -----
    def startAi(self):
        """
        This method starts analog input.
        See document of AioStartAi of API-USBP.
        """
        ret = DLL.AioStartAi(self.Id)
        if ret != 0: #failed
            raise ValueError('AioStartAi failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def startAiSync(self, timeOut):
        """
        This method starts synchronous analog input.
        See document of AioStartAiSync of API-USBP.
        
        :param int timeOut:
            See document of AioStartAiSync of API-USBP.
        """
        ret = DLL.AioStartAiSync(self.Id, timeOut)
        if ret != 0: #failed
            raise ValueError('AioStartAiSync failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def stopAi(self):
        """
        This method stops analog input.
        See document of AioStoptAi of API-USBP.
        """
        ret = DLL.AioStopAi(self.Id)
        if ret != 0: #failed
            raise ValueError('AioStoptAi failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    #----- Getting status -----
    def getAiStatus(self):
        """
        This method returns current analog input status.
        :class:`pyAPISUBP.AIO` has status ID as class attributes
        (e.g. pyAPIUSBP.AIO.AIS_BUSY).
        See document of AioGetAiStatus of API-USBP.
        
        :return:
            See document of AioGetAiStatus of API-USBP.
        """
        status = ctypes.c_long()
        ret = DLL.AioGetAiStatus(self.Id, ctypes.byref(status))
        if ret != 0: #failed
            raise ValueError('AioGetAiStatus failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return status.value
    
    def getAiSamplingCount(self):
        """
        This method returns current number of sampling counts.
        This method works only when device-buffer-mode is set by
        :func:`pyAPISUBP.AIO.getAiTransferMode`
        See document of AioGetAiSamplingCount of API-USBP.
        
        :return:
            See document of AioGetAiSamplingCount of API-USBP.
        """
        count = ctypes.c_long()
        ret = DLL.AioGetAiSamplingCount(self.Id, ctypes.byref(count))
        if ret != 0: #failed
            raise ValueError('AioGetAiSamplingCount failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return count.value
    
    def getAiStopTriggerCount(self):
        """
        This method returns current number of sampling counts when stop trigger
        is input. This method works only when device-buffer-mode is set by
        :func:`pyAPISUBP.AIO.getAiTransferMode`
        See document of AioGetAiStopTriggerCount of API-USBP.
        
        :return:
            See document of AioGetAiStopTriggerCount of API-USBP.
        """
        count = ctypes.c_long()
        ret = DLL.AioGetAiStopTriggerCount(self.Id, ctypes.byref(count))
        if ret != 0: #failed
            raise ValueError('AioGetAiStopTriggerCount failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return count.value
    
    def getAiTransferCount(self):
        """
        This method returns current number of data transferred to user buffer.
        is input. This method works only when user-buffer-mode is set by
        :func:`pyAPISUBP.AIO.getAiTransferMode`
        See document of AioGetAiTransferCount of API-USBP.
        
        :return:
            See document of AioGetAiTransferCount of API-USBP.
        """
        count = ctypes.c_long()
        ret = DLL.AioGetAiTransferCount(self.Id, ctypes.byref(count))
        if ret != 0: #failed
            raise ValueError('AioGetAiTransferCount failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return count.value
    
    def getAiTransferLap(self):
        """
        This method returns how many times user buffer is overwritten.
        This method works only when user-buffer-mode is set by
        :func:`pyAPISUBP.AIO.getAiTransferMode`
        See document of AioGetAiTransferLap of API-USBP.
        
        :return:
            See document of AioGetAiTransferLap of API-USBP.
        """
        lap = ctypes.c_long()
        ret = DLL.AioGetAiTransferLap(self.Id, ctypes.byref(lap))
        if ret != 0: #failed
            raise ValueError('AioGetAiTransferLap failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return lap.value
    
    def getAiStopTriggerTransferCount(self):
        """
        This method returns number of transfers when stop trigger is input.
        This method works only when user-buffer-mode is set by
        :func:`pyAPISUBP.AIO.getAiTransferMode`
        See document of AioGetAiStopTriggerTransferCount of API-USBP.
        
        :return:
            See document of AioGetAiStopTriggerTransferCount of API-USBP.
        """
        count = ctypes.c_long()
        ret = DLL.AioGetAiStopTriggerTransferCount(self.Id, ctypes.byref(count))
        if ret != 0: #failed
            raise ValueError('AioGetAiStopTriggerTransferCount failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return count.value
    
    def getAiRepeatCount(self):
        """
        This method returns crrent repeat count.
        See document of AioGetAiRepeatCount of API-USBP.
        
        :return:
            See document of AioGetAiRepeatCount of API-USBP.
        """
        count = ctypes.c_long()
        ret = DLL.AioGetAiRepeatCount(self.Id, ctypes.byref(count))
        if ret != 0: #failed
            raise ValueError('AioGetAiRepeatCount failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return count.value
    
    #----- Data aquisition -----    
    def getAiSamplingData(self, samplingTimes):

        cSamplingTimes = ctypes.c_long(samplingTimes)
        bufferSize=samplingTimes*self.getAiChannels()
        data = (ctypes.c_long*bufferSize)()
        ret = DLL.AioGetAiSamplingData(self.Id, ctypes.byref(cSamplingTimes), ctypes.byref(data))
        
        #if ret != 0: #failed
        #    raise ValueError('AioGetAiSamplingTimes failed (%s)' % self.getErrorString(ret).decode('sjis'))
           
        return (cSamplingTimes.value, list(data))
    
    def getAiSamplingDataEx(self, samplingTimes):
        """
        This method returns sampling data (in voltage/current) from device
        memory. This method works only when device-buffer-mode is set by
        :func:`pyAPISUBP.AIO.getAiTransferMode`
        See document of AioGetAiSamplingDataEx of API-USBP.
        
        :param int samplingTimes:
            Size of buffer for data receiving.
        :return:
            See document of AioGetAiSamplingDataEx of API-USBP.
        """
        
        cSamplingTimes = ctypes.c_long(samplingTimes)
        bufferSize=samplingTimes*self.getAiChannels()
        data = (ctypes.c_float*bufferSize)()
        ret = DLL.AioGetAiSamplingDataEx(self.Id, ctypes.byref(cSamplingTimes), ctypes.byref(data))
        
        #if ret != 0: #failed
        #    raise ValueError('AioGetAiSamplingTimesEx failed (%s)' % self.getErrorString(ret).decode('sjis'))
        
        return (cSamplingTimes.value, list(data))
    
    #----- Reset -----
    def resetAiStatus(self):
        """
        This method resets analog input status.
        See document of AioResetAiStatus of API-USBP.
        
        :return:
            See document of AioResetAiStatus of API-USBP.
        """
        ret = DLL.AioResetAiStatus(self.Id)
        if ret != 0: #failed
            raise ValueError('AioResetAiStatus failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def resetAiMemory(self):
        """
        This method resets device buffer for analog input.
        This method works only when device-buffer-mode is set by
        :func:`pyAPISUBP.AIO.getAiTransferMode`
        See document of AioResetAiMemory of API-USBP.
        
        :return:
            See document of AioResetAiMemory of API-USBP.
        """
        ret = DLL.AioResetAiMemory(self.Id)
        if ret != 0: #failed
            raise ValueError('AioResetAiMemory failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    
    #------------------------------------------------------------------------------------------
    # Analog Output
    #------------------------------------------------------------------------------------------
    #----- Simple methods -----
    def singleAo(self, channel, data):
        """
        This method writes data to an analog output channel.
        
        :param int channel:
            Channel number.
        :param int data:
            Output data.
        """
        ret = DLL.AioSingleAo(self.Id, channel, data)
        if ret != 0: #failed
            raise ValueError('AioSingleAo failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def singleAoEx(self, channel, data):
        """
        This method writes data (in voltage or current) to an analog output
        channel.
        
        :param int channel:
            Channel number.
        :param int data:
            Output data (in voltage/current).
        """
        ret = DLL.AioSingleAoEx(self.Id, channel, data)
        if ret != 0: #failed
            raise ValueError('AioSingleAoEx failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def multiAo(self, channels, data):
        """
        This method writes data to multiple analog output channels.
        
        :param list channes:
            List of channel numbers.
        :param data:
            List of values (range of values depends on the device).
        """
        cData = (ctypes.c_long*channels)()
        for i in range(channels):
            cData[i] = data[i]
        ret = DLL.AioMultiAo(self.Id, channels, ctypes.byref(cData))
        if ret != 0: #failed
            raise ValueError('AioMultiAo failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def multiAoEx(self, channels, data):
        """
        This method writes data (in voltage or current) to multiple analog
        output channels.
        
        :param list channels:
            List of channel numbers.
        :param data:
            List of values (range of values depends on the device).
        """
        cData = (ctypes.c_float*channels)()
        for i in range(channels):
            cData[i] = data[i]
        ret = DLL.AioMultiAoEx(self.Id, channels, ctypes.byref(cData))
        if ret != 0: #failed
            raise ValueError('AioMultiAoEx failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    #----- Resolution -----
    def getAoResolution(self):
        """
        This function returns resolution of analog output.
        
        :return:
            0 for no-analog-input, 12 for 12bit resolution,
            16 for 16bit resolution.
        """
        resolution = ctypes.c_short()
        ret = DLL.AioGetAoResolution(self.Id, ctypes.byref(resolution))
        if ret != 0: #failed
            raise ValueError('AioGetAoResolution failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return resolution.value
    
    #----- Channels -----
    def setAoChannels(self, channels):
        """
        This function sets number of channels for analog output.
        
        :param int channels:
            Number of channels.
        """
        ret = DLL.AioSetAoChannels(self.Id, channels)
        if ret != 0: #failed
            raise ValueError('AioSetAoChannels failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAoChannels(self):
        """
        This function returns number of channels currently used for analog
        output.
        
        :return:
            Number of channels.
        """
        channels = ctypes.c_short()
        ret = DLL.AioGetAoChannels(self.Id, ctypes.byref(channels))
        if ret != 0: #failed
            raise ValueError('AioGetAoChannels failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return channels.value
    
    def getAoMaxChannels(self):
        """
        This function returns maximum number of analog output channels.
        
        :return:
            Maximum number of analog output channels.
        """
        channels = ctypes.c_short()
        ret = DLL.AioGetAoMaxChannels(self.Id, ctypes.byref(channels))
        if ret != 0: #failed
            raise ValueError('AioGetAoMaxChannels failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return channels.value
    
    #----- Range -----
    def setAoRange(self, channel, range):
        """
        This method sets range of an analog output channel.
        
        :param int channel:
            Analog output channel.
        :param int AiRange:
            Range ID. :class:`pyAPISUBP.AIO` has Range ID as class attributes.
            That is, you can specify range like pyAPISUBP.AIO.P5
        """
        ret = DLL.AioSetAoRange(self.Id, channel, range)
        if ret != 0: #failed
            raise ValueError('AioSetAoRange failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def setAoRangeAll(self, range):
        """
        This method sets range of all analog output channels.
        
        :param int AiRange:
            Range ID. :class:`pyAPISUBP.AIO` has Range ID as class attributes.
            That is, you can specify range like pyAPISUBP.AIO.P5
        """
        ret = DLL.AioGetAoRange(self.Id, range)
        if ret != 0: #failed
            raise ValueError('AioSetAoRangeAll failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAoRange(self, channel):
        """
        This method returns range an analog output channel.
        
        :param int channel:
            Analog output channel.
        :return:
            Range ID. :func:`~pyAPISUBP.AIO.getRangeString` can be used to get 
            range string from range IO.
        """
        range = ctypes.c_short()
        ret = DLL.AioGetAoRange(self.Id, ctypes.byref(range))
        if ret != 0: #failed
            raise ValueError('AioGetAoRange failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return range.value
    
    #----- Transfer mode -----
    def setAoTransferMode(self, mode):
        """
        This method sets data transfer method of analog output.
        
        :param int mode:
            pyAPIUSBP.AIO.CONST_DEVICE_BUFFER for device-buffer-mode,
            pyAPIUSBP.AIO.CONST_USER_BUFFER for user-buffer-mode.
        """
        if not mode in [CONST_DEVICE_BUFFER, CONST_USER_BUFFER]:
            raise ValueError('mode must be CONST_DEVICE_BUFFER or CONST_USER_BUFFER.')
        ret = DLL.AioSetAoTransferMode(self.Id, mode)
        if ret != 0: #failed
            raise ValueError('AioSetAoTransferMode failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAoTransferMode(self):
        """
        This method returns current data transfer method of analog output.
        
        :return:
            pyAPIUSBP.AIO.CONST_DEVICE_BUFFER for device-buffer-mode,
            pyAPIUSBP.AIO.CONST_USER_BUFFER for user-buffer-mode.
        """
        mode = ctypes.c_short()
        ret = DLL.AioGetAoTransferMode(self.Id, ctypes.byref(mode))
        if ret != 0: #failed
            raise ValueError('AioGetAoTransferMode failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return mode.value
    
    #----- Memory type -----
    def setAoMemoryType(self, memType):
        """
        This method sets memory type for data output.
        See document of AioSetAoMemoryType of API-USBP.
        
        :param int memType:
            See document of AioSetAoMemoryType of API-USBP.
        """
        if not memType in [CONST_FIFO, CONST_RING, CONST_MEMORY_NO_REPEAT, CONST_MEMORY_REPEAT]:
            raise ValueError('memType must be CONST_FIFO, CONST_RING, CONST_MEMORY_NO_REPEAT or CONST_MEMORY_REPEAT')
        ret = DLL.AioSetAoMemoryType(self.Id, memType)
        if ret != 0: #failed
            raise ValueError('AioSetAoMemoryType failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAoMemoryType(self):
        """
        This method returns current memory type for data output.
        See document of AioGetAoMemoryType of API-USBP.
        
        :return:
            See document of AioGetAoMemoryType of API-USBP.
        """
        memType = ctypes.c_short()
        ret = DLL.AioGetAoMemoryType(self.Id, ctypes.byref(memType))
        if ret != 0: #failed
            raise ValueError('AioGetAoMemoryType failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return memType.value
    
    #----- Data settings -----
    def setAoSamplingData(self, samplingTimes, data):
        """
        This method sets output data.
        See document of AioSetAoSamplingData of API-USBP.
        
        :param int samplingTimes:
            See document of AioSetAoSamplingData of API-USBP.
        :param int data:
            See document of AioSetAoSamplingData of API-USBP.
        """
        cData = (ctypes.c_long*samplingTimes)()
        for i in range(samplingTimes):
            cData[i] = data[i]
        ret = DLL.AioSetAoSamplingData(self.Id, samplingTimes, ctypes.byref(cData))
        if ret != 0: #failed
            raise ValueError('AioSetAoSamplingData failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def setAoSamplingDataEx(self, samplingTimes, data):
        """
        This method sets output data in voltage/current.
        See document of AioSetAoSamplingData of API-USBP.
        
        :param int samplingTimes:
            See document of AioSetAoSamplingData of API-USBP.
        :param int data:
            See document of AioSetAoSamplingData of API-USBP.
        """
        cData = (ctypes.c_float*samplingTimes)()
        for i in range(samplingTimes):
            cData[i] = data[i]
        ret = DLL.AioSetAoSamplingDataEx(self.Id, samplingTimes, ctypes.byref(cData))
        if ret != 0: #failed
            raise ValueError('AioSetAoSamplingDataEx failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAoSamplingTimes(self):
        """
        This method returns current number of sampling times.
        See document of AioGetAoSamplingTimes of API-USBP.
        
        :return:
            See document of AioGetAoSamplingTimes of API-USBP.
        """
        samplingTimes = ctypes.c_long()
        ret = DLL.AioGetAoSamplingTimes(self.Id, ctypes.byref(samplingTimes))
        if ret != 0: #failed
            raise ValueError('AioGetAoSamplingTimes failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return samplingTimes.value
    
    def setAoTransferData(self, dataNumber, buffer):
        """
        This method sets user buffer for analog output.
        This method works This method works only when user-buffer-mode is set by
        :func:`pyAPISUBP.AIO.getAoTransferMode`.
        See document of AioSetAoTransferData of API-USBP.
        
        :param int dataNumber:
            See document of AioSetAoTransferData of API-USBP.
        :param ctypes.c_long_array buffer:
            See document of AioSetAoTransferData of API-USBP.
        """
        ret = DLL.AioSetAoTransferData(self.Id, dataNumber, buffer)
        if ret != 0: #failed
            raise ValueError('AioSetAoTransferData failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAoSamplingDataSize(self):
        """
        This method returns data size per sample when user-buffer-mode is set
        by :func:`pyAPISUBP.AIO.getAoTransferMode`.
        See document of AioGetAoSamplingDataSize of API-USBP.
        
        :return:
            See document of AioGetAoSamplingDataSize of API-USBP.
        """
        dataSize = ctypes.c_short()
        ret = DLL.AioGetAoSamplingDataSize(self.Id, ctypes.byref(dataSize))
        if ret != 0: #failed
            raise ValueError('AioGetAoSamplingSize failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return dataSize.value
    
    #----- Clock -----
    def setAoClockType(self, clockType):
        """
        See document of AioSetAoClockType of API-USBP.
        """
        ret = DLL.AioSetAoClockType(self.Id, clockType)
        if ret != 0: #failed
            raise ValueError('AioSetAoClockType failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAoClockType(self, clockType):
        """
        See document of AioGetAoClockType of API-USBP.
        """
        clockType = ctypes.c_short()
        ret = DLL.AioGetAoClockType(self.Id, ctypes.byref(clockType))
        if ret != 0: #failed
            raise ValueError('AioGetAoClockType failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return clockType.value
    
    def setAoSamplingClock(self, clock):
        """
        See document of AioSetAoSamplingClock of API-USBP.
        """
        ret = DLL.AioSetAoSamplingClock(self.Id, clock)
        if ret != 0: #failed
            raise ValueError('AioSetAoSamplingClock failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAoSamplingClock(self):
        """
        See document of AioGetAoSamplingClock of API-USBP.
        """
        clock = ctypes.c_float()
        ret = DLL.AioGetAoSamplingClock(self.Id, ctypes.byref(clock))
        if ret != 0: #failed
            raise ValueError('AioGetAoSamplingClock failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return clock.value
    
    def setAoClockEdge(self, edge):
        """
        See document of AioSetAoClockEdge of API-USBP.
        """
        if not edge in [CONST_FALLING_EDGE, CONST_RISING_EDGE]:
            raise ValueError('Edge must be CONST_FALLING_EDGE or CONST_RISING_EDGE.')
        ret = DLL.AioSetAoClockEdge(self.Id, edge)
        if ret != 0: #failed
            raise ValueError('AioSetAoClockEdge failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAoClockEdge(self):
        """
        See document of AioGetAoClockEdge of API-USBP.
        """
        edge = ctypes.c_short()
        ret = DLL.AioGetAoClockEdge(self.Id, ctypes.byref(edge))
        if ret != 0: #failed
            raise ValueError('AioGetAoClockEdge failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return edge.value
    
    #----- Start trigger -----
    def setAoStartTrigger(self, trigger):
        """
        See document of AioSetAoStartTrigger of API-USBP.
        """
        ret = DLL.AioSetAoStartTrigger(self.Id, trigger)
        if ret != 0: #failed
            raise ValueError('AioSetAoStartTrigger failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAoStartTrigger(self):
        """
        See document of AioGetAoStartTrigger of API-USBP.
        """
        trigger = ctypes.c_short()
        ret = DLL.AioGetAoStartTrigger(self.Id, ctypes.byref(trigger))
        if ret != 0: #failed
            raise ValueError('AioGetAoStartTrigger failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return trigger.value
    
    #----- Stop trigger -----
    def setAoStopTrigger(self, trigger):
        """
        See document of AioSetAoStopTrigger of API-USBP.
        """
        ret = DLL.AioSetAoStopTrigger(self.Id, trigger)
        if ret != 0: #failed
            raise ValueError('AioSetAoStopTrigger failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAoStopTrigger(self):
        """
        See document of AioGetAoStopTrigger of API-USBP.
        """
        trigger = ctypes.c_short()
        ret = DLL.AioGetAoStopTrigger(self.Id, ctypes.byref(trigger))
        if ret != 0: #failed
            raise ValueError('AioGetAoStopTrigger failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return trigger.value
    
    #----- Repeat -----
    def setAoRepeatTimes(self, repeatTimes):
        """
        See document of AioSetAoRepeatTimes of API-USBP.
        """
        ret = DLL.AioSetAoRepeatTimes(self.Id, repeatTimes)
        if ret != 0: #failed
            raise ValueError('AioSetAoRepeatTimes failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAoRepeatTimes(self):
        """
        See document of AioGetAoRepeatTimes of API-USBP.
        """
        repeatTimes = ctypes.c_short()
        ret = DLL.AioGetAoRepeatTimes(self.Id, ctypes.byref(repeatTimes))
        if ret != 0: #failed
            raise ValueError('AioGetAoRepeatTimes failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return repeatTimes.value
    
    #----- Event -----
    def setAoEvent(self, hWnd, event):
        """
        This method sets events for analog output.
        See document of AioSetAoEvent of API-USBP.
        
        :param HWND hWnd:
            See document of AioSetAoEvent of API-USBP.
        :param int event:
            See document of AioSetAoEvent of API-USBP.
        """
        ret = DLL.AioSetAoEvent(self.Id, hWnd, event)
        if ret != 0: #failed
            raise ValueError('AioSetAoEvent failed (%s)' % self.getErrorString(ret).decode('sjis'))
        
    def getAoEvent(self, hWnd):
        """
        This method returns events for analog output.
        See document of AioGetAoEvent of API-USBP.
        
        :return:
            See document of AioGetAoEvent of API-USBP.
        """
        hWnd = ctypes.c_int()
        event = ctypes.c_long()
        ret = DLL.AioGetAoEvent(self.Id, ctypes.byref(hWnd), ctypes.byref(event))
        if ret != 0: #failed
            raise ValueError('AioGetAoEvent failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return (hWnd.value, event.value)
        
    def setAoCallBackProc(self, callbackProc, event, param):
        """
        This method sets callback function for event.
        See document of AioSetAoCallBackProc of API-USBP.
        
        :param ctypes.WINFUNCTYPE callbackProc:
            See document of AioSetAoCallBackProc of API-USBP.
        :param int event:
            See document of AioSetAoCallBackProc of API-USBP.
        :param int param:
            See document of AioSetAoCallBackProc of API-USBP.
        """
        ret = DLL.AioSetAoCallBackProc(self.Id, callbackProc, event, param)
        if ret != 0: #failed
            raise ValueError('AioSetAoCallBackProc Failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def setAoEventSamplingTimes(self, samplingTimes):
        """
        This method sets number of sampling times.
        See document of AioSetAoEventSamplingTimes of API-USBP.
        
        :param int samplingTimes:
            See document of AioSetAoEventSamplingTimes of API-USBP.
        """
        ret = DLL.AioSetAoEventSamplingTimes(self.Id , samplingTimes)
        if ret != 0: #failed
            raise ValueError('AioSetAoEventSamplingTimes Failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAoEventSamplingTimes(self):
        """
        This method returns current number of sampling times.
        See document of AioGetAoEventSamplingTimes of API-USBP.
        
        :return:
            See document of AioGetAoEventSamplingTimes of API-USBP.
        """
        samplingTimes = ctypes.c_long()
        ret = DLL.AioGetAoEventSamplingTimes(self.Id , ctypes.byref(samplingTimes))
        if ret != 0: #failed
            raise ValueError('AioGetAoEventSamplingTimes Failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def setAoEventTransferTimes(self, transferTimes):
        """
        This method sets number of transfer times.
        See document of AioSetAoEventTransferTimes of API-USBP.
        
        :param int transferTimes:
            See document of AioSetAoEventTransferTimes of API-USBP.
        """
        ret = DLL.AioSetAoEventTransferTimes(self.Id , transferTimes)
        if ret != 0: #failed
            raise ValueError('AioSetAoEventTransferTimes Failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getAoEventTransferTimes(self):
        """
        This method returns current number of transfer times.
        See document of AioGetAoEventTransferTimes of API-USBP.
        
        :return:
            See document of AioGetAoEventTransferTimes of API-USBP.
        """
        transferTimes = ctypes.c_long()
        ret = DLL.AioGetAoEventTransferTimes(self.Id , ctypes.byref(transferTimes))
        if ret != 0: #failed
            raise ValueError('AioGetAoEventTransferTimes Failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    #----- Start/Stop -----
    def startAo(self):
        """
        This method starts analog output.
        See document of AioStartAo of API-USBP.
        """
        ret = DLL.AioStartAo(self.Id)
        if ret != 0: #failed
            raise ValueError('AioStartAo failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def stopAo(self):
        """
        This method stops analog output.
        See document of AioStoptAo of API-USBP.
        """
        ret = DLL.AioStopAo(self.Id)
        if ret != 0: #failed
            raise ValueError('AioStopAo failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    #----- States -----
    def getAoStatus(self):
        """
        This method returns current analog input status.
        :class:`pyAPISUBP.AIO` has status ID as class attributes
        (e.g. pyAPIUSBP.AIO.AOS_BUSY).
        See document of AioGetAoStatus of API-USBP.
        
        :return:
            See document of AioGetAoStatus of API-USBP.
        """
        status = ctypes.c_long()
        ret = DLL.AioGetAoStatus(self.Id, ctypes.byref(status))
        if ret != 0: #failed
            raise ValueError('AioGetAoStatus failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return status.value
    
    def getAoSamplingCount(self):
        """
        This method returns current number of sampling counts.
        See document of AioGetAoSamplingCount of API-USBP.
        
        :return:
            See document of AioGetAoSamplingCount of API-USBP.
        """
        count = ctypes.c_long()
        ret = DLL.AioGetAoSamplingCount(self.Id, ctypes.byref(count))
        if ret != 0: #failed
            raise ValueError('AioGetAoSamplingCount failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return count.value
    
    def getAoTransferCount(self):
        """
        This method returns current number of data transferred to user buffer.
        is input. This method works only when user-buffer-mode is set by
        :func:`pyAPISUBP.AIO.getAiTransferMode`
        See document of AioGetAiTransferCount of API-USBP.
        
        :return:
            See document of AioGetAiTransferCount of API-USBP.
        """
        count = ctypes.c_long()
        ret = DLL.AioGetAoTransferCount(self.Id, ctypes.byref(count))
        if ret != 0: #failed
            raise ValueError('AioGetAoTransferCount failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return count.value
    
    def getAoRepeatCount(self):
        """
        This method returns crrent repeat count.
        See document of AioGetAoRepeatCount of API-USBP.
        
        :return:
            See document of AioGetAoRepeatCount of API-USBP.
        """
        count = ctypes.c_long()
        ret = DLL.AioGetAoRepeatCount(self.Id, ctypes.byref(count))
        if ret != 0: #failed
            raise ValueError('AioGetAoRepeatCount failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return count.value
    
    
    #----- Reset -----
    def resetAoStatus(self):
        """
        This method resets analog output status.
        See document of AioResetAoStatus of API-USBP.
        
        :return:
            See document of AioResetAoStatus of API-USBP.
        """
        ret = DLL.AioResetAoStatus(self.Id)
        if ret != 0: #failed
            raise ValueError('AioResetAoStatus failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def resetAoMemory(self):
        """
        This method resets device buffer for analog input.
        See document of AioResetAoMemory of API-USBP.
        
        :return:
            See document of AioResetAoMemory of API-USBP.
        """
        ret = DLL.AioResetAoMemory(self.Id)
        if ret != 0: #failed
            raise ValueError('AioResetAoMemory failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    
    #------------------------------------------------------------------------------------------
    # Digital Input
    #------------------------------------------------------------------------------------------
    #----- Simple methods -----
    def inputDiBit(self, bit):
        """
        This method reads a logical input bit.
        
        :param int bitNo:
            Logical input bit number.
        :return:
            Data (0 or 1).
        """
        data = ctypes.c_short()
        ret = DLL.AioInputDiBit(self.Id, bit, ctypes.byref(data))
        if ret != 0: #failed
            raise ValueError('AioInputDiBit failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return data.value
    
    def inputDiByte(self, port):
        """
        This method reads 1 byte data from a logical input port.
        
        :param int portNo:
            Logical input port number. The first port of the device is 0,
            and the following ports are 1, 2, 3...
            Use :func:`~pyAPIUSBP.DIO.getMaxPorts` to query how many 
            ports your device has.
        :return:
            Data (0-255).
        """
        data = ctypes.c_short()
        ret = DLL.AioInputDiByte(self.Id, port, ctypes.byref(data))
        if ret != 0: #failed
            raise ValueError('AioInputDiByte failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return data.value
    
    def setDiFilter(self, bit, value):
        """
        This method sets filter for digital input.
        See ocument of AioSetDiFilter of API-USBP.
        
        :param int bit:
            See ocument of AioSetDiFilter of API-USBP.
        :param float value:
            See ocument of AioSetDiFilter of API-USBP.
        """
        ret = DLL.AioSetDiFilter(self.Id, bit, value)
        if ret != 0: #failed
            raise ValueError('AioSetDiFilter failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getDiFilter(self, bit):
        """
        This method returns current filter setting for digital input.
        See ocument of AioGetDiFilter of API-USBP.
        
        :param int bit:
            See ocument of AioGetDiFilter of API-USBP.
        :return:
            See ocument of AioGetDiFilter of API-USBP.
        """
        value = ctypes.c_short()
        ret = DLL.AioGetDiFilter(self.Id, bit, ctypes.byref(value))
        if ret != 0: #failed
            raise ValueError('AioGetDiFilter failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return value.value
    
    
    #------------------------------------------------------------------------------------------
    # Digital Output
    #------------------------------------------------------------------------------------------
    #----- Simple methods -----
    def outputDoBit(self, bit, data):
        """
        This method writes 0 or 1 to a logical output bit.
        
        :param int bitNo:
            Logical output bit number.
        :param int data:
            Data to be written. Note that the value must be 0 or 1.
        """
        ret = DLL.AioOutputDoBit(self.Id, bit, data)
        if ret != 0: #failed
            raise ValueError('AioOutputDoBit failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def outputDoByte(self, port, data):
        """
        This method writes 1 byte data to a logical output port.
        
        :param int portNo:
            Logical output port number.
        :param int data:
            Data to be written. Note that the value must be between 0 to 255
            (unsigned byte).
        """
        ret = DLL.AioOutputDoByte(self.Id, port, data)
        if ret != 0: #failed
            raise ValueError('AioOutputDoByte failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    
    #------------------------------------------------------------------------------------------
    # Digital IO Direcction
    #------------------------------------------------------------------------------------------
    #----- DIO direction -----
    def setDioDirection(self, direction):
        """
        This method sets direction of digital I/O port.
        See document of AioSetDioDirection of API-USBP.
        
        :param int direction:
            See document of AioSetDioDirection of API-USBP.
        """
        ret = DLL.AioSetDioDirection(self.Id, direction)
        if ret != 0: #failed
            raise ValueError('AioSetDioDirection failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getDioDirection(self):
        """
        This method returns current direction of digital I/O port.
        See document of AioGetDioDirection of API-USBP.
        
        :param int direction:
            See document of AioGetDioDirection of API-USBP.
        """
        direction = ctypes.c_long()
        ret = DLL.AioGetDioDirection(self.Id, ctypes.byref(direction))
        if ret != 0: #failed
            raise ValueError('AioGetDioDirection failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return direction.value
    
    
    #------------------------------------------------------------------------------------------
    # Counter
    #------------------------------------------------------------------------------------------
    #----- Channel -----
    def getCntMaxChannels(self):
        """
        See document of AioGetCntMaxChannels of API-USBP.
        """
        channels = ctypes.c_short()
        ret = DLL.AioGetCntMaxChannels(self.Id, ctypes.byref(channels))
        if ret != 0: #failed
            raise ValueError('AioGetCntMaxChannels failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return channels.value
    
    #----- Mode -----
    def setCntComparisonMode(self, channel, mode):
        """
        See document of AioSetCntComparisonMode of API-USBP.
        """
        ret = DLL.AioSetCntComparisonMode(self.Id, channel, mode)
        if ret != 0: #failed
            raise ValueError('AioSetCntComparisonMode failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getCntComparisonMode(self, channel):
        """
        See document of AioGetCntComparisonMode of API-USBP.
        """
        mode = ctypes.c_short()
        ret = DLL.AioGetCntComparisonMode(self.Id, ctypes.byref(mode))
        if ret != 0: #failed
            raise ValueError('AioGetCntComparisonMode failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return mode.value
    
    #----- Preset -----
    def setCntPresetReg(self, channel, presetNumber, presetData, flag):
        """
        See document of AioSetCntPresetReg of API-USBP.
        """
        cPresetData = (ctypes.c_long*presetNumber)()
        for i in range(presetNumber):
            cPresetData[i] = presetData[i]
        ret = DLL.AioSetCntPresetReg(self.Id, channel, presetNumber, ctypes.byref(cPresetData), flag)
        if ret != 0: #failed
            raise ValueError('AioSetCntPresetReg failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    #----- Comparison count -----
    #def setCntComparisonReg(self, channel, presetNumber, comparisonData, flag):
    #    """
    #    See document of AioSetCntComparisonReg of API-USBP.
    #    """
    #    cComparisonData = (ctypes.c_long*presetNumber)()
    #    for i in range(presetNumber):
    #        cComparisonData[i] = presetData[i]
    #    ret = DLL.AioSetCntComparisonReg(self.Id, channel, presetNumber, ctypes.byref(cComparisonData), flag)
    #    if ret != 0: #failed
    #        raise ValueError('AioSetCntComparisonReg failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    #----- Clock -----
    def setCntInputSignal(self, channel, inputSignal):
        """
        See document of AioSetCntInputSignal of API-USBP.
        """  
        ret = DLL.AioSetCntInputSignal(self.Id, channel, inputSignal)
        if ret != 0: #failed
            raise ValueError('AioSetCntInputSignal failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getCntInputSignal(self):
        """
        See document of AioGetCntInputSignal of API-USBP.
        """    
        inputSignal = ctypes.c_short()
        ret = DLL.AioGetCntInputSignal(self.Id, ctypes.byref(inputSignal))
        if ret != 0: #failed
            raise ValueError('AioGetCntInputSignal failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return inputSignal.value
    
    #----- Event -----
    def setCntEvent(self, channel, hWnd, event):
        """
        See document of AioSetCntEvent of API-USBP.
        """
        ret = DLL.AioSetCntEvent(self.Id, channel, hWnd, event)
        if ret != 0: #failed
            raise ValueError('AioSetCntEvent failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getCntEvent(self, channel):
        """
        See document of AioGetCntEvent of API-USBP.
        """
        hWnd = ctypes.c_int()
        event = ctypes.c_long()
        ret = DLL.AioGetCntEvent(self.Id, channel, ctypes.byref(hWnd), ctypes.byref(event))
        if ret != 0: #failed
            raise ValueError('AioGetCntEvent failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return (hWnd.value, event.value)
    
    def setCntCallBackProc(self, channel, callbackProc, event, param):
        """
        See document of AioSetCntCallBackProc of API-USBP.
        """
        ret = DLL.AioSetCntCallBackProc(self.Id, channel, callbackProc, event, param)
        if ret != 0: #failed
            raise ValueError('AioSetCntCallBackProc Failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    #----- Filter -----
    def setCntFilter(self, channel, signal, value):
        """
        See document of AioSetCntFilter of API-USBP.
        """
        ret = DLL.AioSetCntFilter(self.Id, channel, signal, value)
        if ret != 0: #failed
            raise ValueError('AioSetCntFilter failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getCntFilter(self, channel, signal):
        """
        See document of AioGetCntFilter of API-USBP.
        """
        value = ctypes.c_short()
        ret = DLL.AioGetCntFilter(self.Id, channel, signal, ctypes.byref(value))
        if ret != 0: #failed
            raise ValueError('AioGetCntFilter failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return value.value
    
    #----- Start/Stop -----
    def startCnt(self, channel):
        """
        See document of AioStartCnt of API-USBP.
        """
        ret = DLL.AioStartCnt(self.Id, channel)
        if ret != 0: #failed
            raise ValueError('AioStartCnt failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def stopCnt(self, channel):
        """
        See document of AioStopCnt of API-USBP.
        """
        ret = DLL.AioStopCnt(self.Id, channel)
        if ret != 0: #failed
            raise ValueError('AioStopCnt failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def presetCnt(self, channel, presetData):
        """
        See document of AioPresetCnt of API-USBP.
        """
        ret = DLL.AioPresetCnt(self.Id, channel, presetData)
        if ret != 0: #failed
            raise ValueError('AioPresetCnt failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    #----- Status -----
    def getCntStatus(self, channel):
        """
        See document of AioGetCntStatus of API-USBP.
        """
        status = ctypes.c_short()
        ret = DLL.AioGetCntStatus(self.Id, channel, ctypes.byref(status))
        if ret != 0: #failed
            raise ValueError('AioGetCntStatus failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return status.value
    
    def getCntCount(self, channel):
        """
        See document of AioGetCntCount of API-USBP.
        """
        count = ctypes.c_short()
        ret = DLL.AioGetCntCount(self.Id, channel, ctypes.byref(count))
        if ret != 0: #failed
            raise ValueError('AioGetCntCount failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return count.value
    
    #----- Reset -----
    def resetCntStatus(self, channel, status):
        """
        See document of AioResetCntStatus of API-USBP.
        """
        ret = DLL.AioResetCntStatus(self.Id, channel, status)
        if ret != 0: #failed
            raise ValueError('AioResetCntStatus failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    
    #------------------------------------------------------------------------------------------
    # Timer
    #------------------------------------------------------------------------------------------
    #----- Interval timer -----
    def setTmEvent(self, timerId, hWnd, event):
        """
        See document of AioSetTmEvent of API-USBP.
        """
        ret = DLL.AioSetTmEvent(self.Id, timerId, hWnd, event)
        if ret != 0: #failed
            raise ValueError('AioSetTmEvent failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def getTmEvent(self, timerId):
        """
        See document of AioGetTmEvent of API-USBP.
        """
        hWnd = ctypes.c_int()
        event = ctypes.c_long()
        ret = DLL.AioGetTmEvent(self.Id, timerId, ctypes.byref(hWnd), ctypes.byref(event))
        if ret != 0: #failed
            raise ValueError('AioGetTmEvent failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return (hWnd.value, event.value)
    
    def setTmCallBackProc(self, timerId, callbackProc, event, param):
        """
        See document of AioSetTmCallBackProc of API-USBP.
        """
        ret = DLL.AioSetTmCallBackProc(self.Id, timerId, callbackProc, event, param)
        if ret != 0: #failed
            raise ValueError('AioSetTmCallBackProc Failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    def stopTmTimer(self, timerId):
        """
        See document of AioStopTmTimer of API-USBP.
        """
        ret = DLL.AioStopTmTimer( self.Id, timerId )
        if ret != 0: #failed
            raise ValueError('AioStopTmTimer Failed (%s)' % self.getErrorString(ret).decode('sjis'))
    
    
    #------------------------------------------------------------------------------------------
    # Event controller
    #------------------------------------------------------------------------------------------
    #----- Event controller -----
    def setEcuSignal(self, destination, source):
        """
        See document of AioSetEcuSignal of API-USBP.
        """
        ret = DLL.AioSetEcuSignal(self.Id, destination, source)
        if ret != 0: #failed
            raise ValueError('AioSetEcuSignal failed (%s)' % self.getErrorString(ret).decode('sjis'))
        
    def getEcuSignal(self, destination):
        """
        See document of AioGetEcuSignal of API-USBP.
        """
        source = ctypes.c_short()
        ret = DLL.AioGetEcuSignal(self.Id, destination, ctypes.byref(source))
        if ret != 0: #failed
            raise ValueError('AioGetEcuSignal failed (%s)' % self.getErrorString(ret).decode('sjis'))
        return source.value
