import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import DataFrame
import numpy as np
import os
from scipy import signal


def is_continues_frame(last_frame, present_frame):  # if in the same minute, do
    time1 = last_frame.split(':')
    time2 = present_frame.split(':')
    if len(time1[3]) == 2:
        time1[3] = str(0) + time1[3]
    elif len(time1[3]) == 1:
        time1[3] = '0' + '0' + time1[3]
    if len(time2[3]) == 2:
        time2[3] = str(0) + time2[3]
    elif len(time2[3]) == 1:
        time2[3] = '0' + '0' + time2[3]
    last_frame_split = int(time1[2]+time1[3])
    present_frame_split = int(time2[2]+time2[3])

    result = present_frame_split - last_frame_split
    if result > 13:
        return int(result/10)  # gain the No. of Dataframe plugged in
    else:
        return False           # regard it as normal performance

# load dataset and padding into 100hz: for PLC


def PLC_padding_with_last_value(filename1):
    # e.g.               pad train set into 100Hz
    # 11:28:26:667,xxxx           11:28:26:667,xxxx
    # 11:28:26:677,xxxx    ===>   11:28:26:677,xxxx
    # 11:28:26:717,yyyy           11:28:26:687,xxxx
    #                             11:28:26:697,xxxx
    #                             11:28:26:707,xxxx
    #                             11:28:26:717,yyyy
    values = read_csv(filename1, header=0).values  # make csv into np array
    shape = values.shape
    ret_np_array = []
    for i in range(1, shape[0]):  # from 1 to the end of this col
        timeframe = values[i-1, 0].split(':')
        if values[i-1, -1] != values[i, -1]:   # if not the same minute
            med_frame = values[i - 1]
            if len(timeframe[3]) == 2:
                timeframe[3] = str(0) + timeframe[3]
            elif len(timeframe[3]) == 1:
                timeframe[3] = '0' + '0' + timeframe[3]
            med_frame[0] = int(timeframe[0] + timeframe[1] + timeframe[2] + timeframe[3])
            ret_np_array.append(np.array([med_frame[0], med_frame[1], med_frame[2], med_frame[3],
                                          med_frame[4], med_frame[5]]))
        else:                                  # if same minute, then exam if the plug needed
            plug = is_continues_frame(values[i-1, 0], values[i, 0])
            if not plug:
                med_frame = values[i - 1]
                if len(timeframe[3]) == 2:
                    timeframe[3] = str(0) + timeframe[3]
                elif len(timeframe[3]) == 1:
                    timeframe[3] = '0' + '0' + timeframe[3]
                med_frame[0] = int(timeframe[0] + timeframe[1] + timeframe[2] + timeframe[3])
                ret_np_array.append(np.array([med_frame[0], med_frame[1], med_frame[2], med_frame[3],
                                              med_frame[4], med_frame[5]]))

            else:
                for j in range(plug):
                    med_frame = values[i-1]
                    if len(timeframe[3]) == 2:
                        timeframe[3] = str(0) + timeframe[3]
                    elif len(timeframe[3]) == 1:
                        timeframe[3] = str(0) + str(0) + timeframe[3]
                    med_frame[0] = int(timeframe[0]+timeframe[1]+timeframe[2]+timeframe[3])
                    med_frame[0] += int(10*j)
                    ret_np_array.append(np.array([med_frame[0], med_frame[1], med_frame[2], med_frame[3],
                                                  med_frame[4], med_frame[5]]))
                    # ret_np_array[-1][0] = int(timeframe[0]+timeframe[1]+timeframe[2]+timeframe[3])
                    # print(ret_np_array[-1][0])
                    # ret_np_array[-1][0] = ret_np_array[-1][0]+10*(j+1)
    return ret_np_array

'''
first_path = 'D:\CCC\CSS_data\\train\\01-TrainingData-qLua'
second_listdir = os.listdir(first_path)
minute_array = []
for second_path in second_listdir:
    if os.path.isdir(first_path+'\\'+second_path):
        csv_list = os.listdir(first_path+'\\'+second_path+'\\PLC')
        cutter_feats = []
        i = 0
        for csv_file in csv_list:
            i += 1
            filename = first_path+'\\'+second_path+'\\PLC\\'+'plc.csv'
            minute_array = PLC_padding_with_last_value(filename)
            plc_pad = DataFrame(columns=None, data=minute_array)
            plc_pad = signal.resample(plc_pad, num=96000, axis=0)
            np.save(first_path+'\\'+second_path+'\\'+str(i), plc_pad)  # save into npy
'''


def is_continues_minute(last_array, current_array):
    if round(last_array[-1] == round(current_array[-1])):
        return True
    else:
        return False


def connect_all_plc(filename_plc):
    minute_array = PLC_padding_with_last_value(filename_plc)
    last_array = np.array([1000, 1000])
    test_show = np.array(minute_array)
    list_plc = []
    idx_left = -1
    idx_right = 0
    len_minute_array = len(minute_array)
    for idx_array in minute_array:

        if is_continues_minute(last_array, idx_array) and len_minute_array > idx_right+1:
            pass
        elif idx_left != -1:
            print(idx_left, idx_right)
            print(minute_array[idx_left][-1],minute_array[idx_right][-1])
            list_plc.append(
                np.pad(
                    signal.resample(
                        DataFrame(minute_array[idx_left:idx_right])
                        , num=1980, axis=0)
                    , ((0, 20), (0, 0)), 'constant')
            )
            idx_left = idx_right
        else:
            idx_left = 0
        idx_right += 1
        last_array = idx_array
    print(len_minute_array, idx_right)
    return list_plc


filename_plc1 = 'D:\\CCC\\CSS_data\\train\\01-TrainingData-qLua\\01\\PLC\\plc.csv'
filename_plc2 = 'D:\\CCC\\CSS_data\\train\\01-TrainingData-qLua\\02\\PLC\\plc.csv'
filename_plc3 = 'D:\\CCC\\CSS_data\\train\\01-TrainingData-qLua\\03\\PLC\\plc.csv'
filename_plc4 = 'D:\CCC\CSS_data\\test\\02-TestingData-keD1\\01\PLC\\plc.csv'
filename_plc5 = 'D:\CCC\CSS_data\\test\\02-TestingData-keD1\\02\PLC\\plc.csv'
filename_plc6 = 'D:\CCC\CSS_data\\test\\02-TestingData-keD1\\03\PLC\\plc.csv'
filename_plc7 = 'D:\CCC\CSS_data\\test\\02-TestingData-keD1\\04\PLC\\plc.csv'
filename_plc8 = 'D:\CCC\CSS_data\\test\\02-TestingData-keD1\\05\PLC\\plc.csv'
# #
# list_plc1 = connect_all_plc(filename_plc1)
# list_plc2 = connect_all_plc(filename_plc2)
list_plc3 = connect_all_plc(filename_plc3)
# list_plc4 = connect_all_plc(filename_plc4)
# list_plc5 = connect_all_plc(filename_plc5)
# list_plc6 = connect_all_plc(filename_plc6)
# list_plc7 = connect_all_plc(filename_plc7)
# list_plc8 = connect_all_plc(filename_plc8)
# ret_plc = np.concatenate((np.array(list_plc1), np.array(list_plc2)), axis=0)
# np.save('train_set_plc.npy', ret_plc[:, :, 1:5])
# np.save('train_set_label.npy', ret_plc[:, :, -1])
# ret_plc = np.concatenate((np.array(list_plc1), np.array(list_plc2),
#                           np.array(list_plc3)), axis=0)
ret_plc = np.array(list_plc3)
# np.save('test_set_plc.npy', ret_plc[:, :, 1:5])
np.save('valid_set_label.npy', ret_plc[:, :, -1])
#
# list_plc3 = np.array(list_plc3)
# np.save('valid_set.npy', list_plc3[:, :, 1:5])
# np.save('valid_set_label', list_plc3[:, :, -1])


