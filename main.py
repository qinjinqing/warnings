import pandas as pd
import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
import scipy.signal as sig
import numpy as np


def realtime_warning(data, thresholds, **kwargs):
    """
    预警算法
    Args:
        data(pandas DataFrame): 各个监测项目的数据
        thresholds(pandas DataFrame): 各个监测项目的阈值
        **kwargs: python信号处理函数find_peak中其他参数，optional
    Returns:
      warnings (pandas DataFrame): columns= ['warning_source(监测项目)', 'warning_level(预警等级)', 'start_time', 'end_time', 'maxima(峰值)']

    """
    warnings = pd.DataFrame([], None, columns=['warning_source', 'warning_level', 'start_time', 'end_time', 'maxima'])
    i = 1
    if isinstance(data, pd.DataFrame) and isinstance(thresholds, pd.DataFrame):
        while i <= len(data.columns):
            single_data = data.iloc[:, i]  # 单个监测项目的数据
            data_column_name = data.columns.values[i - 1]
            single_threshold = thresholds.loc[data_column_name]
            threshold = threshold_rank(single_threshold)
            single_kwargs = data_match_kwargs(data_column_name, kwargs)
            for index in threshold:
                peaks, property = sig.find_peaks(single_data, height=threshold[index],
                                                 threshold=single_kwargs['threshold'],
                                                 distance=single_kwargs['distance'],
                                                 prominence=single_kwargs['prominence'], width=single_kwargs['width'],
                                                 wlen=single_kwargs['wlen'], rel_height=single_kwargs['rel_height'],
                                                 plateau_size=single_kwargs['plateau_size'])
                insert_warning = create_warnings(data_column_name, threshold[index], peaks, property)
            warnings = pd.concat([warnings, insert_warning], ignore_index=True)
        return warnings

    else:
        error = "data或thresholds非pandas Dataframe"
        return error


def data_match_thresholds(column_name, thresholds):
    """
    监测数据与阈值数据的名称匹配
    Args:
        column_name(string):
        thresholds(pandas dataframe):

    Returns:
        match_column_name(string)

    """

def create_warnings(source, rank, peaks, property):
    """
    获取监测项目超出预警值得预警信息
    Args:
        source(string):预警的监测项目
        rank(string):预警等级
        peaks(ndarray):波峰
        property(dict):

    Returns:
         warnings(dataframe):监测项目在超出指定预警值的预警信息
    """
    warning = []
    for tt in peaks:
        warning_data = [source, rank, peaks[tt], peaks[tt], property['peak_heights'][tt]]
        warning.append(warning_data)
    columns = ['warning_source', 'warning_level', 'start_time', 'end_time', 'maxima']
    warnings = pd.DataFrame(warning_data, columns=columns)
    return warnings

def threshold_rank(threshold_data):
    """
    将单个监测项目阈值数据分成各个预警等级对应的阈值
    Args:
        threshold_data(series):

    Returns:
        threshold(list)
    """
    threshold = threshold_data
    return threshold


def data_match_kwargs(column_name, kwargs):
    """

    Args:
        column_name(string):
        kwargs(dict):

    Returns:
        single_kwargs:
    """
    single_kwargs = {'threshold': None, 'distance': None, 'prominence': None, 'width': None, 'wlen': None,
                     'rel_height': None, 'plateau_size': None}
    return single_kwargs


if __name__ == '__main__':
    x = np.linspace(0, 6 * np.pi, 1000)
    x = np.sin(x) + 0.6 * np.sin(2.6 * x)
    """
    y = electrocardiogram()[2000:4000]
    peaks, property = sig.find_peaks(y, height=0, distance=None)
    print(property)
    plt.plot(y)
    plt.plot(peaks, y[peaks], "x")
    plt.plot(np.zeros_like(y), "--", color="gray")
    plt.show()
    data = {
        '性别': ['男', '女', '女', '男', '男'],
        '姓名': ['小明', '小红', '小芳', '大黑', '张三'],
        '年龄': [20, 21, 25, 24, 29]}
    df = pd.DataFrame(data, index=['one', 'two', 'three', 'four', 'five'], columns=['姓名', '性别', '年龄', '职业'])
    print(df.columns.values[0])
    
    """
    peaks, property = sig.find_peaks(-x, height=0)

    results_half = sig.peak_widths(x, peaks, rel_height=0.5)
    print(peaks)
    print(results_half)
    plt.plot(x)
    plt.plot(peaks, x[peaks], "x")
    plt.hlines(*results_half[1:], color="C2")
    #plt.hlines(*results_full[1:], color="C3")
    plt.show()

