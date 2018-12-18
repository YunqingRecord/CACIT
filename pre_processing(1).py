import matplotlib.pyplot as plt
from pandas import read_csv
import numpy as np
from scipy import signal
import os


def is_continues_frame(last_frame,present_frame):
    last_frame_split = last_frame.split(':')
    present_frame_split = present_frame.split(':')
    #if 
# load dataset and padding into 100hz: for PLC 
def PLC_padding_with_last_value(filename):

#e.g.               pad train set into 100Hz
#11:28:26:667,xxxx           11:28:26:667,xxxx
#11:28:26:677,xxxx    ===>   11:28:26:677,xxxx
#11:28:26:717,yyyy           11:28:26:687,xxxx
##                           11:28:26:697,xxxx
##                           11:28:26:707,xxxx
##                           11:28:26:717,yyyy
    values = read_csv(filename, header=0).values #make csv into np array
    shape = values.shape
    ret_np_array = []
    for i in range(1,shape[0]):
        #if is_continues_frame(values[i-1,0],values[i,0]): #cmp adjacent frames
        ret_np_array.append(values[i,0].split(':')[1])
    return ret_np_array

#os.listdir()
# filename = '01-TrainingData-qLua/03/PLC/plc.csv'
# minute_array = padding_with_last_value(filename)

'''
Sensor data pre-processing
1.methods
    (1) lowpass filter. <??4300 Hertz    --> value need to be well tuned..  
    (2) under-sampling with n1 time-step --> get n1 new signal   
    (3) short time Fourier transform     --> get n2 spectrum data where n2 is resoluted by the sliding windows' overlap size
    (4) set label
2.results
    we get n1*n2 spectrum data(per minute) for training.
    result saving format : np array
'''


def Sensor_pre_processing(filename):
    values = read_csv(filename, header=0).values
    Fs = 25600  # the sampling frequency
    step_sampling = 2
    under_sample_freq = int(Fs/step_sampling) # 欠采样后的 采样频率
    list_pxx = []
    for idx in range(4):
        print(filename, ': vibration_dim_',idx)
        array_dim = values[:, idx]
        #滤波上限;fs/2/2 ->除以2为采样定理+，除以2代表欠采样间隔 #暂时未采样
        filter_ceiling_freq = int(Fs/2/step_sampling)
        # filter_floor_freq = 0 #not used in lowpass filter
        # using Butterworth filter (order 8 FIR)
        b, a = signal.butter(8, filter_ceiling_freq*2/Fs, 'lowpass')
    
        #filtedData = signal.filtfilt(b, a, array_dim)
        filtedData = array_dim
        # the length of the windowing segments
        # NFFT = 25600
        # #reduce to int(25600/8) to save memory
        NFFT = int(25600/8)
        # use Short-time FT to calculate spectrogram
        #overlap = (25600/8*33 - 25600 - overlap))/32
        # => 32*overlap = (25600/8*33 - 25600 - overlap)
        overlap_n = int((25600/8*33 - 25600)/33)

        fig, (ax1, ax) = plt.subplots(2)
        Pxx, freqs, bins, im = ax.specgram(filtedData, NFFT=NFFT, Fs=Fs, noverlap=overlap_n)
        # ax1.plot(Pxx)
        # plt.show()
        # print(csv_num)
        # plt.savefig('res3_sensor_specgram/test_specgram'+str(csv_num)+'.png')
        plt.close()
        shape_Pxx = Pxx.shape
        #padding to 2000 time-step
        if(shape_Pxx[1]<2000):
            Pxx = np.pad(Pxx, ((0, 0), (0, 2000 - shape_Pxx[1])), 'constant')
        else:
            Pxx = Pxx[:, 0:2000]
        list_pxx.append(Pxx)
    # 4 dim spectrogram
    return np.array(list_pxx)


first_path = 'D:\CCC\CSS_data\\test\\02-TestingData-keD1'
second_listdir = os.listdir(first_path)
Sensor_spectrogram_feats = []
for second_path in second_listdir:
    if os.path.isdir(first_path+'/'+second_path):
        csv_list = os.listdir(first_path+'/'+second_path+'/Sensor')
        cutter_feats = []
        i = 0
        for csv_file in csv_list:
            i += 1
            filename = first_path+'/'+second_path+'/Sensor/'+csv_file
            spectrogram_ch4 = np.swapaxes(Sensor_pre_processing(filename), 0, 2)  # keep(time-step,feature,channel)format
            #cutter_feats.append(spectrogram_ch4)
            print(spectrogram_ch4.shape)
            print()
            #spectrogram_ch4.astype('float16')
            Sensor_spectrogram_feats.append(spectrogram_ch4.astype('float16'))
            #np.save(first_path+'/'+second_path+'/'+str(i), spectrogram_ch4) #save into npy
np.save('test_set_sensor.npy', np.array(Sensor_spectrogram_feats)) #save into npy
#Sensor_spectrogram_feats = np.array(Sensor_spectrogram_feats)


# link all plc data into 1 feature array
'''
path1 = '01-TrainingData-qLua/01/'
plc_arr = np.load(path1+'PLC/1.npy')
shape_x = plc_arr.shape
#plc 需要padding到timestep=2000, sensor已经在函数中padding过了
plc_arr = np.pad(plc_arr, ((0, 2000-shape_x[0]),(0,0)),'constant')

#按序把每分钟的数据 连接起来, 这里连接两个相同的作为演示
#实际处理时用循环读取处理前两个刀片的所有数据
list_plc = []
list_plc.append(plc_arr) #plc_arr.shape = [2000,5] 4 features + 1 label
list_plc.append(plc_arr)
ret_plc = np.array(list_plc)
np.save('train_set_plc.npy', ret_plc[:,:,0:4])#记得删除时间轴
np.save('trann_set_label.npy', ret_plc[:,:,-1])
'''
'''
other testcode to plot 1-dim  fft res

for csv_num in range(1,14):
    print(csv_num)
    filename = '01-TrainingData-qLua/02/Sensor/'+str(csv_num)+'.csv'
    values = read_csv(filename, header=0).values
    for columns in range(1,2):
        dir_name = 'sensor2_all_vib'+ str(columns)
        for i in range(1,60):
            Fs = 25600 #采样频率
            #print(i)
            seconds = i+1 
            length = Fs*seconds # N (window length for FFT) 越长，频域分辨率越高
            vibration1 = values[0:length,columns]
            freq_domain = np.arange(length)*Fs/length # 频率轴

            sp = np.fft.fft(vibration1)
            abs_sp = abs(sp) #amplitude combine real&imaginary freq domain

            plt.figure()
            plt.plot(freq_domain[int(length/100):int(length/2)],abs_sp[int(length/100):int(length/2)])
            plt.xlabel("frequence")
            plt.ylabel("amplitude")
            plt.title(str(csv_num))
            try:
                plt.savefig(dir_name+'/'+str(csv_num)+'sensor_fft'+str(seconds)+'.png')
            except:
                os.mkdir(dir_name)
                plt.savefig(dir_name+'/'+str(csv_num)+'sensor_fft'+str(seconds)+'.png')
            plt.close()
'''