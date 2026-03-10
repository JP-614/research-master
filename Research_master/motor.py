import re
def degree(name):
    if(name=='上島L'): #左手義手
        hk=35   #4指の最大屈曲時のモータ指令値
        hs=80   #4指の最大伸展時のモータ指令値
        wk=40   #手首の最大回内時のモータ指令値
        ws=150  #手首の最大回外時のモータ指令値
        ek=23   #肘の最大屈曲時のモータ指令値
        ess=80  #肘の最大伸展時のモータ指令値
        tk=180  #拇指の最大対立時
        ts=100  #拇指の最大並立時
        degree=[hk,hs,wk,ws,ek,ess,tk,ts]
        return degree
    
    if(name=='上島R'): #右手義手
        hk=140   #4指の最大屈曲時のモータ指令値
        hs=77   #4指の最大伸展時のモータ指令値
        wk=150   #手首の最大回内時のモータ指令値
        ws=40  #手首の最大回外時のモータ指令値
        ek=23   #肘の最大屈曲時のモータ指令値
        ess=80  #肘の最大伸展時のモータ指令値
        tk=10  #拇指の最大対立時
        ts=100  #拇指の最大並立時
        degree=[hk,hs,wk,ws,ek,ess,tk,ts]
        return degree

    elif(name=='Kさん'): #左手義手
        hk=30   #4指の最大屈曲時のモータ指令値
        hs=150  #4指の最大伸展時のモータ指令値
        wk=60   #手首の最大回内時のモータ指令値
        ws=160  #手首の最大回外時のモータ指令値
        ek=60   #肘の最大屈曲時のモータ指令値
        ess=120 #肘の最大伸展時のモータ指令値
        tk=170  #拇指の最大対立時
        ts=70   #拇指の最大並立時
        degree=[hk,hs,wk,ws,ek,ess,tk,ts]
        return degree
    
    elif(name=='Iさん'): #右手義手
        hk=150  #4指の最大屈曲時のモータ指令値
        hs=30   #4指の最大伸展時のモータ指令値
        wk=254  #手首の最大回内時のモータ指令値
        ws=40   #手首の最大回外時のモータ指令値
        ek=70   #肘の最大屈曲時のモータ指令値
        ess=160 #肘の最大伸展時のモータ指令値
        tk=30   #拇指の最大対立時
        ts=170  #拇指の最大並立時        
        degree=[hk,hs,wk,ws,ek,ess,tk,ts]
        return degree
    
    elif(name=='NRさん'): #左手義手
        hk=80   #4指の最大屈曲時のモータ指令値
        hs=220  #4指の最大伸展時のモータ指令値
        wk=50   #手首の最大回内時のモータ指令値
        ws=250  #手首の最大回外時のモータ指令値
        ek=50   #肘の最大屈曲時のモータ指令値
        ess=180 #肘の最大伸展時のモータ指令値
        tk=200  #拇指の最大対立時
        ts=50   #拇指の最大並立時
        degree=[hk,hs,wk,ws,ek,ess,tk,ts]
        return degree
    
    elif(name=='NSさん'): #右手義手
        hk=160  #4指の最大屈曲時のモータ指令値
        hs=30   #4指の最大伸展時のモータ指令値
        wk=210  #手首の最大回内時のモータ指令値
        ws=10   #手首の最大回外時のモータ指令値
        ek=70   #肘の最大屈曲時のモータ指令値
        ess=160 #肘の最大伸展時のモータ指令値
        tk=30   #拇指の最大対立時
        ts=180  #拇指の最大並立時
        degree=[hk,hs,wk,ws,ek,ess,tk,ts]
        return degree
    
    else:
        hk=100  #4指の最大屈曲時のモータ指令値
        hs=100  #4指の最大伸展時のモータ指令値
        wk=100  #手首の最大回内時のモータ指令値
        ws=100  #手首の最大回外時のモータ指令値
        ek=100  #肘の最大屈曲時のモータ指令値
        ess=100 #肘の最大伸展時のモータ指令値
        tk=100  #拇指の最大対立時
        ts=100  #拇指の最大並立時
        degree=[hk,hs,wk,ws,ek,ess,tk,ts]
        return degree

def motormove(movename):
    posturename = movename.split(']')[1]
    posture=re.split('(..)',posturename)[1::2]
    if(posture[0]=='安静'):
        hand=0
    elif(posture[0]=='握り'):
        hand=1
    elif(posture[0]=='開き'):
        hand=-1

    if(posture[1]=='安静'):
        wrist=0
    elif(posture[1]=='回内'):
        wrist=1
    elif(posture[1]=='回外'):
        wrist=-1

    if(posture[2]=='安静'):
        elbow=0
    elif(posture[2]=='屈曲'):
        elbow=1
    elif(posture[2]=='伸展'):
        elbow=-1

    return [hand,wrist,elbow]