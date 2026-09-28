"""Run RQ2 on the original systems (data1-data8)."""

import time
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.ensemble import RandomForestRegressor
from skmultiflow import drift_detection
from skmultiflow.drift_detection import ADWIN
import datetime
import os
import logging
import sys
# sys.path.append("/mnt/sdaDisk/xzz/dhda")
from DaLModels import DaL_Regressor
from base_model.basemodels import build_incremental_model

seed = 43
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")

def data_path(relative_path):
    """Resolve a dataset path relative to the repository code directory."""
    return os.path.join(DATA_DIR, *relative_path.split("/"))


work_dir = os.path.join(PROJECT_ROOT, "result", "rq2-v3")


def create_required_dirs(parent_dir):
    """
    检查并创建指定目录下所需的子目录

    参数:
        parent_dir (str): 父目录路径
    """
    # 需要检查和创建的子目录列表
    required_dirs = ['csv', 'log', 'figures', 'sort', 'stats']

    # 确保父目录存在
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
        print(f"已创建父目录: {parent_dir}")

    # 检查并创建每个子目录
    for dir_name in required_dirs:
        dir_path = os.path.join(parent_dir, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"已创建子目录: {dir_path}")
        else:
            print(f"子目录已存在: {dir_path}")


create_required_dirs(work_dir)
env_list1 = [data_path("data1/sac_2.csv"),
             data_path("data1/sac_4.csv"),
             data_path("data1/sac_5.csv"),
             data_path("data1/sac_6.csv"),
             data_path("data1/sac_7.csv"),
             data_path("data1/sac_8.csv"),
             data_path("data1/sac_9.csv")]

env_list2 = [data_path("data2/x264_0.csv"),
             data_path("data2/x264_1.csv"),
             data_path("data2/x264_2.csv"),
             data_path("data2/x264_3.csv"),
             data_path("data2/x264_4.csv"),
             data_path("data2/x264_5.csv"),
             data_path("data2/x264_6.csv"),
             data_path("data2/x264_7.csv"),
             data_path("data2/x264_8.csv"),
             data_path("data2/x264_9.csv"),
             data_path("data2/x264_10.csv"),
             data_path("data2/x264_11.csv"),
             data_path("data2/x264_12.csv"),
             data_path("data2/x264_13.csv"),
             data_path("data2/x264_14.csv"),
             data_path("data2/x264_15.csv"),
             data_path("data2/x264_16.csv"),
             data_path("data2/x264_17.csv"),
             data_path("data2/x264_18.csv"),
             data_path("data2/x264_19.csv"),
             data_path("data2/x264_20.csv"), ]

env_list3 = [data_path("data3/storm-obj1_feature6.csv"),
             data_path("data3/storm-obj1_feature7.csv"),
             data_path("data3/storm-obj1_feature8.csv"),
             data_path("data3/storm-obj2_feature9.csv"),
             data_path("data3/storm-obj2_feature6.csv"),
             data_path("data3/storm-obj1_feature7.csv"),
             data_path("data3/storm-obj2_feature8.csv"),
             data_path("data3/storm-obj2_feature9.csv")]

env_list4 = [data_path("data4/spear_0.csv"),
             data_path("data4/spear_1.csv"),
             data_path("data4/spear_2.csv"),
             data_path("data4/spear_3.csv"),
             data_path("data4/spear_4.csv"),
             data_path("data4/spear_5.csv"),
             data_path("data4/spear_6.csv"),
             data_path("data4/spear_7.csv"),
             data_path("data4/spear_8.csv"),
             data_path("data4/spear_9.csv")]

env_list5 = [data_path("data5/sqlite_10.csv"),
             data_path("data5/sqlite_11.csv"),
             data_path("data5/sqlite_16.csv"),
             data_path("data5/sqlite_17.csv"),
             data_path("data5/sqlite_18.csv"),
             data_path("data5/sqlite_19.csv"),
             data_path("data5/sqlite_44.csv"),
             data_path("data5/sqlite_45.csv"),
             data_path("data5/sqlite_52.csv"),
             data_path("data5/sqlite_59.csv"),
             data_path("data5/sqlite_73.csv"),
             data_path("data5/sqlite_79.csv"),
             data_path("data5/sqlite_94.csv"),
             data_path("data5/sqlite_96.csv"),
             data_path("data5/sqlite_97.csv")]

env_list6 = [data_path("data6/nginx1.csv"),
             data_path("data6/nginx2.csv"),
             data_path("data6/nginx3.csv"),
             data_path("data6/nginx4.csv")]

env_list7 = [data_path("data7/exastencils1.csv"),
             data_path("data7/exastencils2.csv"),
             data_path("data7/exastencils3.csv"),
             data_path("data7/exastencils4.csv")]

env_list8 = [data_path("data8/deeparch1.csv"),
             data_path("data8/deeparch2.csv"),
             data_path("data8/deeparch3.csv")
             ]

env_list9 = [data_path("data9/data_12threads_100connections_10s_False.csv"), data_path("data9/data_12threads_100connections_10s_true.csv"), data_path("data9/data_12threads_100connections_120s_False.csv"), data_path("data9/data_12threads_100connections_120s_true.csv"), data_path("data9/data_12threads_100connections_30s_true.csv"), data_path("data9/data_12threads_100connections_60s_False.csv"), data_path("data9/data_12threads_300connections_10s_False.csv"), data_path("data9/data_12threads_300connections_120s_False.csv"), data_path("data9/data_12threads_300connections_30s_False.csv"), data_path("data9/data_12threads_300connections_30s_true.csv"), data_path("data9/data_12threads_300connections_60s_False.csv"), data_path("data9/data_4threads_100connections_10s_False.csv"), data_path("data9/data_4threads_10connections_10s_False.csv"), data_path("data9/data_4threads_10connections_10s_true.csv"), data_path("data9/data_4threads_10connections_120s_False.csv"), data_path("data9/data_4threads_10connections_120s_true.csv"), data_path("data9/data_4threads_10connections_30s_False.csv"), data_path("data9/data_4threads_10connections_30s_true.csv"), data_path("data9/data_4threads_10connections_60s_False.csv"), data_path("data9/data_4threads_10connections_60s_true.csv"), data_path("data9/data_4threads_300connections_10s_true.csv"), data_path("data9/data_4threads_300connections_120s_False.csv"), data_path("data9/data_4threads_300connections_120s_true.csv"), data_path("data9/data_4threads_300connections_30s_False.csv"), data_path("data9/data_4threads_300connections_60s_False.csv"), data_path("data9/data_4threads_300connections_60s_true.csv"), data_path("data9/data_8threads_10connections_10s_False.csv"), data_path("data9/data_8threads_10connections_10s_true.csv"), data_path("data9/data_8threads_10connections_120s_False.csv"), data_path("data9/data_8threads_10connections_120s_true.csv"), data_path("data9/data_8threads_10connections_30s_False.csv"), data_path("data9/data_8threads_10connections_30s_true.csv"), data_path("data9/data_8threads_10connections_60s_False.csv"), data_path("data9/data_8threads_10connections_60s_true.csv"), data_path("data9/data_8threads_300connections_10s_False.csv")]

env_list10 = [data_path("data10/factor_data_120s_20tbls_200ksizes_01rw_16thds_size.csv"), data_path("data10/factor_data_120s_20tbls_500ksizes_10rw_32thds_size.csv"), data_path("data10/factor_data_120s_20tbls_50ksizes_01rw_32thds_size.csv"), data_path("data10/factor_data_120s_40tbls_500ksizes_10rw_16thds_size.csv"), data_path("data10/factor_data_120s_60tbls_100ksizes_01rw_16thds_size.csv"), data_path("data10/factor_data_120s_60tbls_200ksizes_55rw_32thds_size.csv"), data_path("data10/factor_data_120s_60tbls_500ksizes_55rw_16thds_size.csv"), data_path("data10/factor_data_120s_60tbls_500ksizes_55rw_32thds_size.csv"), data_path("data10/factor_data_120s_80tbls_100ksizes_55rw_16thds_size.csv"), data_path("data10/factor_data_120s_80tbls_200ksizes_10rw_16thds_size.csv"), data_path("data10/factor_data_120s_80tbls_200ksizes_55rw_16thds_size.csv"), data_path("data10/factor_data_120s_80tbls_50ksizes_01rw_16thds_size.csv"), data_path("data10/factor_data_180s_20tbls_100ksizes_01rw_16thds_size.csv"), data_path("data10/factor_data_180s_20tbls_100ksizes_55rw_32thds_size.csv"), data_path("data10/factor_data_180s_20tbls_200ksizes_01rw_32thds_size.csv"), data_path("data10/factor_data_180s_20tbls_200ksizes_10rw_16thds_size.csv"), data_path("data10/factor_data_180s_40tbls_100ksizes_01rw_32thds_size.csv"), data_path("data10/factor_data_180s_40tbls_100ksizes_10rw_16thds_size.csv"), data_path("data10/factor_data_180s_40tbls_500ksizes_55rw_16thds_size.csv"), data_path("data10/factor_data_180s_40tbls_50ksizes_10rw_16thds_size.csv"), data_path("data10/factor_data_180s_60tbls_100ksizes_10rw_32thds_size.csv"), data_path("data10/factor_data_180s_60tbls_200ksizes_10rw_16thds_size.csv"), data_path("data10/factor_data_180s_60tbls_500ksizes_55rw_16thds_size.csv"), data_path("data10/factor_data_180s_80tbls_200ksizes_01rw_16thds_size.csv"), data_path("data10/factor_data_180s_80tbls_500ksizes_55rw_32thds_size.csv"), data_path("data10/factor_data_30s_20tbls_100ksizes_01rw_32thds_size.csv"), data_path("data10/factor_data_30s_20tbls_100ksizes_10rw_16thds_size.csv"), data_path("data10/factor_data_30s_20tbls_500ksizes_01rw_16thds_size.csv"), data_path("data10/factor_data_30s_40tbls_200ksizes_55rw_16thds_size.csv"), data_path("data10/factor_data_30s_40tbls_500ksizes_10rw_16thds_size.csv"), data_path("data10/factor_data_30s_40tbls_50ksizes_01rw_16thds_size.csv"), data_path("data10/factor_data_30s_40tbls_50ksizes_10rw_16thds_size.csv"), data_path("data10/factor_data_30s_60tbls_500ksizes_10rw_16thds_size.csv"), data_path("data10/factor_data_30s_60tbls_500ksizes_10rw_32thds_size.csv"), data_path("data10/factor_data_30s_60tbls_50ksizes_55rw_16thds_size.csv"), data_path("data10/factor_data_30s_80tbls_100ksizes_55rw_16thds_size.csv"), data_path("data10/factor_data_30s_80tbls_100ksizes_55rw_32thds_size.csv"), data_path("data10/factor_data_30s_80tbls_200ksizes_01rw_32thds_size.csv"), data_path("data10/factor_data_30s_80tbls_500ksizes_55rw_32thds_size.csv"), data_path("data10/factor_data_30s_80tbls_50ksizes_01rw_16thds_size.csv"), data_path("data10/factor_data_30s_80tbls_50ksizes_10rw_32thds_size.csv"), data_path("data10/factor_data_60s_40tbls_100ksizes_01rw_16thds_size.csv"), data_path("data10/factor_data_60s_40tbls_100ksizes_10rw_32thds_size.csv"), data_path("data10/factor_data_60s_40tbls_200ksizes_55rw_32thds_size.csv"), data_path("data10/factor_data_60s_40tbls_500ksizes_01rw_16thds_size.csv"), data_path("data10/factor_data_60s_40tbls_500ksizes_10rw_32thds_size.csv"), data_path("data10/factor_data_60s_40tbls_500ksizes_55rw_16thds_size.csv"), data_path("data10/factor_data_60s_40tbls_50ksizes_55rw_16thds_size.csv"), data_path("data10/factor_data_60s_60tbls_100ksizes_10rw_16thds_size.csv"), data_path("data10/factor_data_60s_60tbls_200ksizes_55rw_32thds_size.csv"), data_path("data10/factor_data_60s_60tbls_500ksizes_01rw_16thds_size.csv"), data_path("data10/factor_data_60s_60tbls_500ksizes_55rw_16thds_size.csv"), data_path("data10/factor_data_60s_80tbls_100ksizes_01rw_32thds_size.csv"), data_path("data10/factor_data_60s_80tbls_100ksizes_10rw_16thds_size.csv"), data_path("data10/factor_data_60s_80tbls_200ksizes_55rw_16thds_size.csv"), data_path("data10/factor_data_60s_80tbls_200ksizes_55rw_32thds_size.csv"), data_path("data10/factor_data_60s_80tbls_500ksizes_01rw_16thds_size.csv"), data_path("data10/factor_data_60s_80tbls_500ksizes_10rw_16thds_size.csv"), data_path("data10/factor_data_60s_80tbls_500ksizes_55rw_16thds_size.csv"), data_path("data10/factor_data_60s_80tbls_50ksizes_01rw_16thds_size.csv"), data_path("data10/factor_data_60s_80tbls_50ksizes_01rw_32thds_size.csv")]

env_list11 = [data_path("data11/1280x720-gradient_blue-red-Grayscale-JPG-16.csv"), data_path("data11/1280x720-gradient_blue-red-Grayscale-JPG-8.csv"), data_path("data11/1280x720-gradient_blue-red-Grayscale-PNG-16.csv"), data_path("data11/1280x720-gradient_blue-red-Grayscale-PNG-8.csv"), data_path("data11/1280x720-gradient_blue-red-Grayscale-WEBP-16.csv"), data_path("data11/1280x720-gradient_blue-red-Grayscale-WEBP-8.csv"), data_path("data11/1280x720-gradient_blue-red-sRGB-JPG-16.csv"), data_path("data11/1280x720-gradient_blue-red-sRGB-JPG-8.csv"), data_path("data11/1280x720-gradient_blue-red-sRGB-PNG-16.csv"), data_path("data11/1280x720-gradient_blue-red-sRGB-PNG-8.csv"), data_path("data11/1280x720-gradient_blue-red-sRGB-WEBP-16.csv"), data_path("data11/1280x720-gradient_blue-red-sRGB-WEBP-8.csv"), data_path("data11/1280x720-gradient_blue-red-sRGBA-JPG-16.csv"), data_path("data11/1280x720-gradient_blue-red-sRGBA-JPG-8.csv"), data_path("data11/1280x720-gradient_blue-red-sRGBA-PNG-16.csv"), data_path("data11/1280x720-gradient_blue-red-sRGBA-PNG-8.csv"), data_path("data11/1280x720-gradient_blue-red-sRGBA-WEBP-16.csv"), data_path("data11/1280x720-gradient_blue-red-sRGBA-WEBP-8.csv"), data_path("data11/1280x720-pattern_checkerboard-Grayscale-JPG-16.csv"), data_path("data11/1280x720-pattern_checkerboard-Grayscale-JPG-8.csv"), data_path("data11/1280x720-pattern_checkerboard-Grayscale-PNG-16.csv"), data_path("data11/1280x720-pattern_checkerboard-Grayscale-PNG-8.csv"), data_path("data11/1280x720-pattern_checkerboard-Grayscale-WEBP-16.csv"), data_path("data11/1280x720-pattern_checkerboard-Grayscale-WEBP-8.csv"), data_path("data11/1280x720-pattern_checkerboard-sRGB-JPG-16.csv"), data_path("data11/1280x720-pattern_checkerboard-sRGB-JPG-8.csv"), data_path("data11/1280x720-pattern_checkerboard-sRGB-PNG-16.csv"), data_path("data11/1280x720-pattern_checkerboard-sRGB-PNG-8.csv"), data_path("data11/1280x720-pattern_checkerboard-sRGB-WEBP-16.csv"), data_path("data11/1280x720-pattern_checkerboard-sRGB-WEBP-8.csv"), data_path("data11/1280x720-pattern_checkerboard-sRGBA-JPG-16.csv"), data_path("data11/1280x720-pattern_checkerboard-sRGBA-JPG-8.csv"), data_path("data11/1280x720-pattern_checkerboard-sRGBA-PNG-16.csv"), data_path("data11/1280x720-pattern_checkerboard-sRGBA-PNG-8.csv"), data_path("data11/1280x720-pattern_checkerboard-sRGBA-WEBP-16.csv"), data_path("data11/1280x720-pattern_checkerboard-sRGBA-WEBP-8.csv"), data_path("data11/1280x720-plasma_fractal-Grayscale-JPG-16.csv"), data_path("data11/1280x720-plasma_fractal-Grayscale-JPG-8.csv"), data_path("data11/1280x720-plasma_fractal-Grayscale-PNG-16.csv"), data_path("data11/1280x720-plasma_fractal-Grayscale-PNG-8.csv"), data_path("data11/1280x720-plasma_fractal-Grayscale-WEBP-16.csv"), data_path("data11/1280x720-plasma_fractal-Grayscale-WEBP-8.csv"), data_path("data11/1280x720-plasma_fractal-sRGB-JPG-16.csv"), data_path("data11/1280x720-plasma_fractal-sRGB-JPG-8.csv"), data_path("data11/1280x720-plasma_fractal-sRGB-PNG-16.csv"), data_path("data11/1280x720-plasma_fractal-sRGB-PNG-8.csv"), data_path("data11/1280x720-plasma_fractal-sRGB-WEBP-16.csv"), data_path("data11/1280x720-plasma_fractal-sRGB-WEBP-8.csv"), data_path("data11/1280x720-plasma_fractal-sRGBA-JPG-16.csv"), data_path("data11/1280x720-plasma_fractal-sRGBA-JPG-8.csv"), data_path("data11/1280x720-plasma_fractal-sRGBA-PNG-16.csv"), data_path("data11/1280x720-plasma_fractal-sRGBA-PNG-8.csv"), data_path("data11/1280x720-plasma_fractal-sRGBA-WEBP-16.csv"), data_path("data11/1280x720-plasma_fractal-sRGBA-WEBP-8.csv"), data_path("data11/1920x1080-gradient_blue-red-Grayscale-JPG-16.csv"), data_path("data11/1920x1080-gradient_blue-red-Grayscale-JPG-8.csv"), data_path("data11/1920x1080-gradient_blue-red-Grayscale-PNG-16.csv"), data_path("data11/1920x1080-gradient_blue-red-Grayscale-PNG-8.csv"), data_path("data11/1920x1080-gradient_blue-red-Grayscale-WEBP-16.csv"), data_path("data11/1920x1080-gradient_blue-red-Grayscale-WEBP-8.csv"), data_path("data11/1920x1080-gradient_blue-red-sRGB-JPG-16.csv"), data_path("data11/1920x1080-gradient_blue-red-sRGB-JPG-8.csv"), data_path("data11/1920x1080-gradient_blue-red-sRGB-PNG-16.csv"), data_path("data11/1920x1080-gradient_blue-red-sRGB-PNG-8.csv"), data_path("data11/1920x1080-gradient_blue-red-sRGB-WEBP-16.csv"), data_path("data11/1920x1080-gradient_blue-red-sRGB-WEBP-8.csv"), data_path("data11/1920x1080-gradient_blue-red-sRGBA-JPG-16.csv"), data_path("data11/1920x1080-gradient_blue-red-sRGBA-JPG-8.csv"), data_path("data11/1920x1080-gradient_blue-red-sRGBA-PNG-16.csv"), data_path("data11/1920x1080-gradient_blue-red-sRGBA-PNG-8.csv"), data_path("data11/1920x1080-gradient_blue-red-sRGBA-WEBP-16.csv"), data_path("data11/1920x1080-gradient_blue-red-sRGBA-WEBP-8.csv"), data_path("data11/1920x1080-pattern_checkerboard-Grayscale-JPG-16.csv"), data_path("data11/1920x1080-pattern_checkerboard-Grayscale-JPG-8.csv"), data_path("data11/1920x1080-pattern_checkerboard-Grayscale-PNG-16.csv"), data_path("data11/1920x1080-pattern_checkerboard-Grayscale-PNG-8.csv"), data_path("data11/1920x1080-pattern_checkerboard-Grayscale-WEBP-16.csv"), data_path("data11/1920x1080-pattern_checkerboard-Grayscale-WEBP-8.csv"), data_path("data11/1920x1080-pattern_checkerboard-sRGB-JPG-16.csv"), data_path("data11/1920x1080-pattern_checkerboard-sRGB-JPG-8.csv"), data_path("data11/1920x1080-pattern_checkerboard-sRGB-PNG-16.csv"), data_path("data11/1920x1080-pattern_checkerboard-sRGB-PNG-8.csv"), data_path("data11/1920x1080-pattern_checkerboard-sRGB-WEBP-16.csv"), data_path("data11/1920x1080-pattern_checkerboard-sRGB-WEBP-8.csv"), data_path("data11/1920x1080-pattern_checkerboard-sRGBA-JPG-16.csv"), data_path("data11/1920x1080-pattern_checkerboard-sRGBA-JPG-8.csv"), data_path("data11/1920x1080-pattern_checkerboard-sRGBA-PNG-16.csv"), data_path("data11/1920x1080-pattern_checkerboard-sRGBA-PNG-8.csv"), data_path("data11/1920x1080-pattern_checkerboard-sRGBA-WEBP-16.csv"), data_path("data11/1920x1080-pattern_checkerboard-sRGBA-WEBP-8.csv"), data_path("data11/1920x1080-plasma_fractal-Grayscale-JPG-16.csv"), data_path("data11/1920x1080-plasma_fractal-Grayscale-JPG-8.csv"), data_path("data11/1920x1080-plasma_fractal-Grayscale-PNG-16.csv"), data_path("data11/1920x1080-plasma_fractal-Grayscale-PNG-8.csv"), data_path("data11/1920x1080-plasma_fractal-Grayscale-WEBP-16.csv"), data_path("data11/1920x1080-plasma_fractal-Grayscale-WEBP-8.csv"), data_path("data11/1920x1080-plasma_fractal-sRGB-JPG-16.csv"), data_path("data11/1920x1080-plasma_fractal-sRGB-JPG-8.csv"), data_path("data11/1920x1080-plasma_fractal-sRGB-PNG-16.csv"), data_path("data11/1920x1080-plasma_fractal-sRGB-PNG-8.csv"), data_path("data11/1920x1080-plasma_fractal-sRGB-WEBP-16.csv"), data_path("data11/1920x1080-plasma_fractal-sRGB-WEBP-8.csv"), data_path("data11/1920x1080-plasma_fractal-sRGBA-JPG-16.csv"), data_path("data11/1920x1080-plasma_fractal-sRGBA-JPG-8.csv"), data_path("data11/1920x1080-plasma_fractal-sRGBA-PNG-16.csv"), data_path("data11/1920x1080-plasma_fractal-sRGBA-PNG-8.csv"), data_path("data11/1920x1080-plasma_fractal-sRGBA-WEBP-16.csv"), data_path("data11/1920x1080-plasma_fractal-sRGBA-WEBP-8.csv"), data_path("data11/2560x1440-gradient_blue-red-Grayscale-JPG-16.csv"), data_path("data11/2560x1440-gradient_blue-red-Grayscale-JPG-8.csv"), data_path("data11/2560x1440-gradient_blue-red-Grayscale-PNG-16.csv"), data_path("data11/2560x1440-gradient_blue-red-Grayscale-PNG-8.csv"), data_path("data11/2560x1440-gradient_blue-red-Grayscale-WEBP-16.csv"), data_path("data11/2560x1440-gradient_blue-red-Grayscale-WEBP-8.csv"), data_path("data11/2560x1440-gradient_blue-red-sRGB-JPG-16.csv"), data_path("data11/2560x1440-gradient_blue-red-sRGB-JPG-8.csv"), data_path("data11/2560x1440-gradient_blue-red-sRGB-PNG-16.csv"), data_path("data11/2560x1440-gradient_blue-red-sRGB-PNG-8.csv"), data_path("data11/2560x1440-gradient_blue-red-sRGB-WEBP-16.csv"), data_path("data11/2560x1440-gradient_blue-red-sRGB-WEBP-8.csv"), data_path("data11/2560x1440-gradient_blue-red-sRGBA-JPG-16.csv"), data_path("data11/2560x1440-gradient_blue-red-sRGBA-JPG-8.csv"), data_path("data11/2560x1440-gradient_blue-red-sRGBA-PNG-16.csv"), data_path("data11/2560x1440-gradient_blue-red-sRGBA-PNG-8.csv"), data_path("data11/2560x1440-gradient_blue-red-sRGBA-WEBP-16.csv"), data_path("data11/2560x1440-gradient_blue-red-sRGBA-WEBP-8.csv"), data_path("data11/2560x1440-pattern_checkerboard-Grayscale-JPG-16.csv"), data_path("data11/2560x1440-pattern_checkerboard-Grayscale-JPG-8.csv"), data_path("data11/2560x1440-pattern_checkerboard-Grayscale-PNG-16.csv"), data_path("data11/2560x1440-pattern_checkerboard-Grayscale-PNG-8.csv"), data_path("data11/2560x1440-pattern_checkerboard-Grayscale-WEBP-16.csv"), data_path("data11/2560x1440-pattern_checkerboard-Grayscale-WEBP-8.csv"), data_path("data11/2560x1440-pattern_checkerboard-sRGB-JPG-16.csv"), data_path("data11/2560x1440-pattern_checkerboard-sRGB-JPG-8.csv"), data_path("data11/2560x1440-pattern_checkerboard-sRGB-PNG-16.csv"), data_path("data11/2560x1440-pattern_checkerboard-sRGB-PNG-8.csv"), data_path("data11/2560x1440-pattern_checkerboard-sRGB-WEBP-16.csv"), data_path("data11/2560x1440-pattern_checkerboard-sRGB-WEBP-8.csv"), data_path("data11/2560x1440-pattern_checkerboard-sRGBA-JPG-16.csv"), data_path("data11/2560x1440-pattern_checkerboard-sRGBA-JPG-8.csv"), data_path("data11/2560x1440-pattern_checkerboard-sRGBA-PNG-16.csv"), data_path("data11/2560x1440-pattern_checkerboard-sRGBA-PNG-8.csv"), data_path("data11/2560x1440-pattern_checkerboard-sRGBA-WEBP-16.csv"), data_path("data11/2560x1440-pattern_checkerboard-sRGBA-WEBP-8.csv"), data_path("data11/2560x1440-plasma_fractal-Grayscale-JPG-16.csv"), data_path("data11/2560x1440-plasma_fractal-Grayscale-JPG-8.csv"), data_path("data11/2560x1440-plasma_fractal-Grayscale-PNG-16.csv"), data_path("data11/2560x1440-plasma_fractal-Grayscale-PNG-8.csv"), data_path("data11/2560x1440-plasma_fractal-Grayscale-WEBP-16.csv"), data_path("data11/2560x1440-plasma_fractal-Grayscale-WEBP-8.csv"), data_path("data11/2560x1440-plasma_fractal-sRGB-JPG-16.csv"), data_path("data11/2560x1440-plasma_fractal-sRGB-JPG-8.csv"), data_path("data11/2560x1440-plasma_fractal-sRGB-PNG-16.csv"), data_path("data11/2560x1440-plasma_fractal-sRGB-PNG-8.csv"), data_path("data11/2560x1440-plasma_fractal-sRGB-WEBP-16.csv"), data_path("data11/2560x1440-plasma_fractal-sRGB-WEBP-8.csv"), data_path("data11/2560x1440-plasma_fractal-sRGBA-JPG-16.csv"), data_path("data11/2560x1440-plasma_fractal-sRGBA-JPG-8.csv"), data_path("data11/2560x1440-plasma_fractal-sRGBA-PNG-16.csv"), data_path("data11/2560x1440-plasma_fractal-sRGBA-PNG-8.csv"), data_path("data11/2560x1440-plasma_fractal-sRGBA-WEBP-16.csv"), data_path("data11/2560x1440-plasma_fractal-sRGBA-WEBP-8.csv"), data_path("data11/3840x2160-gradient_blue-red-Grayscale-JPG-16.csv"), data_path("data11/3840x2160-gradient_blue-red-Grayscale-JPG-8.csv"), data_path("data11/3840x2160-gradient_blue-red-Grayscale-PNG-16.csv"), data_path("data11/3840x2160-gradient_blue-red-Grayscale-PNG-8.csv"), data_path("data11/3840x2160-gradient_blue-red-Grayscale-WEBP-16.csv"), data_path("data11/3840x2160-gradient_blue-red-Grayscale-WEBP-8.csv"), data_path("data11/3840x2160-gradient_blue-red-sRGB-JPG-16.csv"), data_path("data11/3840x2160-gradient_blue-red-sRGB-JPG-8.csv"), data_path("data11/3840x2160-gradient_blue-red-sRGB-PNG-16.csv"), data_path("data11/3840x2160-gradient_blue-red-sRGB-PNG-8.csv"), data_path("data11/3840x2160-gradient_blue-red-sRGB-WEBP-16.csv"), data_path("data11/3840x2160-gradient_blue-red-sRGB-WEBP-8.csv"), data_path("data11/3840x2160-gradient_blue-red-sRGBA-JPG-16.csv"), data_path("data11/3840x2160-gradient_blue-red-sRGBA-JPG-8.csv"), data_path("data11/3840x2160-gradient_blue-red-sRGBA-PNG-16.csv"), data_path("data11/3840x2160-gradient_blue-red-sRGBA-PNG-8.csv"), data_path("data11/3840x2160-gradient_blue-red-sRGBA-WEBP-16.csv"), data_path("data11/3840x2160-gradient_blue-red-sRGBA-WEBP-8.csv"), data_path("data11/3840x2160-pattern_checkerboard-Grayscale-JPG-16.csv"), data_path("data11/3840x2160-pattern_checkerboard-Grayscale-JPG-8.csv"), data_path("data11/3840x2160-pattern_checkerboard-Grayscale-PNG-16.csv"), data_path("data11/3840x2160-pattern_checkerboard-Grayscale-PNG-8.csv"), data_path("data11/3840x2160-pattern_checkerboard-Grayscale-WEBP-16.csv"), data_path("data11/3840x2160-pattern_checkerboard-Grayscale-WEBP-8.csv"), data_path("data11/3840x2160-pattern_checkerboard-sRGB-JPG-16.csv"), data_path("data11/3840x2160-pattern_checkerboard-sRGB-JPG-8.csv"), data_path("data11/3840x2160-pattern_checkerboard-sRGB-PNG-16.csv"), data_path("data11/3840x2160-pattern_checkerboard-sRGB-PNG-8.csv"), data_path("data11/3840x2160-pattern_checkerboard-sRGB-WEBP-16.csv"), data_path("data11/3840x2160-pattern_checkerboard-sRGB-WEBP-8.csv"), data_path("data11/3840x2160-pattern_checkerboard-sRGBA-JPG-16.csv"), data_path("data11/3840x2160-pattern_checkerboard-sRGBA-JPG-8.csv"), data_path("data11/3840x2160-pattern_checkerboard-sRGBA-PNG-16.csv"), data_path("data11/3840x2160-pattern_checkerboard-sRGBA-PNG-8.csv"), data_path("data11/3840x2160-pattern_checkerboard-sRGBA-WEBP-16.csv"), data_path("data11/3840x2160-pattern_checkerboard-sRGBA-WEBP-8.csv"), data_path("data11/3840x2160-plasma_fractal-Grayscale-JPG-16.csv"), data_path("data11/3840x2160-plasma_fractal-Grayscale-JPG-8.csv"), data_path("data11/3840x2160-plasma_fractal-Grayscale-PNG-16.csv"), data_path("data11/3840x2160-plasma_fractal-Grayscale-PNG-8.csv"), data_path("data11/3840x2160-plasma_fractal-Grayscale-WEBP-16.csv"), data_path("data11/3840x2160-plasma_fractal-Grayscale-WEBP-8.csv"), data_path("data11/3840x2160-plasma_fractal-sRGB-JPG-16.csv"), data_path("data11/3840x2160-plasma_fractal-sRGB-JPG-8.csv"), data_path("data11/3840x2160-plasma_fractal-sRGB-PNG-16.csv"), data_path("data11/3840x2160-plasma_fractal-sRGB-PNG-8.csv"), data_path("data11/3840x2160-plasma_fractal-sRGB-WEBP-16.csv"), data_path("data11/3840x2160-plasma_fractal-sRGB-WEBP-8.csv"), data_path("data11/3840x2160-plasma_fractal-sRGBA-JPG-16.csv"), data_path("data11/3840x2160-plasma_fractal-sRGBA-JPG-8.csv"), data_path("data11/3840x2160-plasma_fractal-sRGBA-PNG-16.csv"), data_path("data11/3840x2160-plasma_fractal-sRGBA-PNG-8.csv"), data_path("data11/3840x2160-plasma_fractal-sRGBA-WEBP-16.csv"), data_path("data11/3840x2160-plasma_fractal-sRGBA-WEBP-8.csv")]
env_list12 = [data_path("data12/1280x720-noise_gauss-avif-cmyk-float-none-1.csv"), data_path("data12/1280x720-noise_gauss-avif-cmyk-float-none-3.csv"), data_path("data12/1280x720-noise_gauss-avif-cmyk-float-transparent-1.csv"), data_path("data12/1280x720-noise_gauss-avif-cmyk-float-transparent-3.csv"), data_path("data12/1280x720-noise_gauss-avif-cmyk-uchar-none-1.csv"), data_path("data12/1280x720-noise_gauss-avif-cmyk-uchar-none-3.csv"), data_path("data12/1280x720-noise_gauss-avif-cmyk-uchar-transparent-1.csv"), data_path("data12/1280x720-noise_gauss-avif-cmyk-uchar-transparent-3.csv"), data_path("data12/1280x720-noise_gauss-avif-srgb-float-none-1.csv"), data_path("data12/1280x720-noise_gauss-avif-srgb-float-none-3.csv"), data_path("data12/1280x720-noise_gauss-avif-srgb-float-transparent-1.csv"), data_path("data12/1280x720-noise_gauss-avif-srgb-float-transparent-3.csv"), data_path("data12/1280x720-noise_gauss-avif-srgb-uchar-none-1.csv"), data_path("data12/1280x720-noise_gauss-avif-srgb-uchar-none-3.csv"), data_path("data12/1280x720-noise_gauss-avif-srgb-uchar-transparent-1.csv"), data_path("data12/1280x720-noise_gauss-avif-srgb-uchar-transparent-3.csv"), data_path("data12/1280x720-noise_gauss-jpg-cmyk-float-none-1.csv"), data_path("data12/1280x720-noise_gauss-jpg-cmyk-float-none-3.csv"), data_path("data12/1280x720-noise_gauss-jpg-cmyk-float-transparent-1.csv"), data_path("data12/1280x720-noise_gauss-jpg-cmyk-float-transparent-3.csv"), data_path("data12/1280x720-noise_gauss-jpg-cmyk-uchar-none-1.csv"), data_path("data12/1280x720-noise_gauss-jpg-cmyk-uchar-none-3.csv"), data_path("data12/1280x720-noise_gauss-jpg-cmyk-uchar-transparent-1.csv"), data_path("data12/1280x720-noise_gauss-jpg-cmyk-uchar-transparent-3.csv"), data_path("data12/1280x720-noise_gauss-jpg-srgb-float-none-1.csv"), data_path("data12/1280x720-noise_gauss-jpg-srgb-float-none-3.csv"), data_path("data12/1280x720-noise_gauss-jpg-srgb-float-transparent-1.csv"), data_path("data12/1280x720-noise_gauss-jpg-srgb-float-transparent-3.csv"), data_path("data12/1280x720-noise_gauss-jpg-srgb-uchar-none-1.csv"), data_path("data12/1280x720-noise_gauss-jpg-srgb-uchar-none-3.csv"), data_path("data12/1280x720-noise_gauss-jpg-srgb-uchar-transparent-1.csv"), data_path("data12/1280x720-noise_gauss-jpg-srgb-uchar-transparent-3.csv"), data_path("data12/1280x720-solid_black-avif-cmyk-float-none-1.csv"), data_path("data12/1280x720-solid_black-avif-cmyk-float-none-3.csv"), data_path("data12/1280x720-solid_black-avif-cmyk-float-transparent-1.csv"), data_path("data12/1280x720-solid_black-avif-cmyk-float-transparent-3.csv"), data_path("data12/1280x720-solid_black-avif-cmyk-uchar-none-1.csv"), data_path("data12/1280x720-solid_black-avif-cmyk-uchar-none-3.csv"), data_path("data12/1280x720-solid_black-avif-cmyk-uchar-transparent-1.csv"), data_path("data12/1280x720-solid_black-avif-cmyk-uchar-transparent-3.csv"), data_path("data12/1280x720-solid_black-avif-srgb-float-none-1.csv"), data_path("data12/1280x720-solid_black-avif-srgb-float-none-3.csv"), data_path("data12/1280x720-solid_black-avif-srgb-float-transparent-1.csv"), data_path("data12/1280x720-solid_black-avif-srgb-float-transparent-3.csv"), data_path("data12/1280x720-solid_black-avif-srgb-uchar-none-1.csv"), data_path("data12/1280x720-solid_black-avif-srgb-uchar-none-3.csv"), data_path("data12/1280x720-solid_black-avif-srgb-uchar-transparent-1.csv"), data_path("data12/1280x720-solid_black-avif-srgb-uchar-transparent-3.csv"), data_path("data12/1280x720-solid_black-jpg-cmyk-float-none-1.csv"), data_path("data12/1280x720-solid_black-jpg-cmyk-float-none-3.csv"), data_path("data12/1280x720-solid_black-jpg-cmyk-float-transparent-1.csv"), data_path("data12/1280x720-solid_black-jpg-cmyk-float-transparent-3.csv"), data_path("data12/1280x720-solid_black-jpg-cmyk-uchar-none-1.csv"), data_path("data12/1280x720-solid_black-jpg-cmyk-uchar-none-3.csv"), data_path("data12/1280x720-solid_black-jpg-cmyk-uchar-transparent-1.csv"), data_path("data12/1280x720-solid_black-jpg-cmyk-uchar-transparent-3.csv"), data_path("data12/1280x720-solid_black-jpg-srgb-float-none-1.csv"), data_path("data12/1280x720-solid_black-jpg-srgb-float-none-3.csv"), data_path("data12/1280x720-solid_black-jpg-srgb-float-transparent-1.csv"), data_path("data12/1280x720-solid_black-jpg-srgb-float-transparent-3.csv"), data_path("data12/1280x720-solid_black-jpg-srgb-uchar-none-1.csv"), data_path("data12/1280x720-solid_black-jpg-srgb-uchar-none-3.csv"), data_path("data12/1280x720-solid_black-jpg-srgb-uchar-transparent-1.csv"), data_path("data12/1280x720-solid_black-jpg-srgb-uchar-transparent-3.csv"), data_path("data12/1280x720-texture_worley-avif-cmyk-float-none-1.csv"), data_path("data12/1280x720-texture_worley-avif-cmyk-float-none-3.csv"), data_path("data12/1280x720-texture_worley-avif-cmyk-float-transparent-1.csv"), data_path("data12/1280x720-texture_worley-avif-cmyk-float-transparent-3.csv"), data_path("data12/1280x720-texture_worley-avif-cmyk-uchar-none-1.csv"), data_path("data12/1280x720-texture_worley-avif-cmyk-uchar-none-3.csv"), data_path("data12/1280x720-texture_worley-avif-cmyk-uchar-transparent-1.csv"), data_path("data12/1280x720-texture_worley-avif-cmyk-uchar-transparent-3.csv"), data_path("data12/1280x720-texture_worley-avif-srgb-float-none-1.csv"), data_path("data12/1280x720-texture_worley-avif-srgb-float-none-3.csv"), data_path("data12/1280x720-texture_worley-avif-srgb-float-transparent-1.csv"), data_path("data12/1280x720-texture_worley-avif-srgb-float-transparent-3.csv"), data_path("data12/1280x720-texture_worley-avif-srgb-uchar-none-1.csv"), data_path("data12/1280x720-texture_worley-avif-srgb-uchar-none-3.csv"), data_path("data12/1280x720-texture_worley-avif-srgb-uchar-transparent-1.csv"), data_path("data12/1280x720-texture_worley-avif-srgb-uchar-transparent-3.csv"), data_path("data12/1280x720-texture_worley-jpg-cmyk-float-none-1.csv"), data_path("data12/1280x720-texture_worley-jpg-cmyk-float-none-3.csv"), data_path("data12/1280x720-texture_worley-jpg-cmyk-float-transparent-1.csv"), data_path("data12/1280x720-texture_worley-jpg-cmyk-float-transparent-3.csv"), data_path("data12/1280x720-texture_worley-jpg-cmyk-uchar-none-1.csv"), data_path("data12/1280x720-texture_worley-jpg-cmyk-uchar-none-3.csv"), data_path("data12/1280x720-texture_worley-jpg-cmyk-uchar-transparent-1.csv"), data_path("data12/1280x720-texture_worley-jpg-cmyk-uchar-transparent-3.csv"), data_path("data12/1280x720-texture_worley-jpg-srgb-float-none-1.csv"), data_path("data12/1280x720-texture_worley-jpg-srgb-float-none-3.csv"), data_path("data12/1280x720-texture_worley-jpg-srgb-float-transparent-1.csv"), data_path("data12/1280x720-texture_worley-jpg-srgb-float-transparent-3.csv"), data_path("data12/1280x720-texture_worley-jpg-srgb-uchar-none-1.csv"), data_path("data12/1280x720-texture_worley-jpg-srgb-uchar-none-3.csv"), data_path("data12/1280x720-texture_worley-jpg-srgb-uchar-transparent-1.csv"), data_path("data12/1280x720-texture_worley-jpg-srgb-uchar-transparent-3.csv"), data_path("data12/1920x1080-noise_gauss-avif-cmyk-float-none-1.csv"), data_path("data12/1920x1080-noise_gauss-avif-cmyk-float-none-3.csv"), data_path("data12/1920x1080-noise_gauss-avif-cmyk-float-transparent-1.csv"), data_path("data12/1920x1080-noise_gauss-avif-cmyk-float-transparent-3.csv"), data_path("data12/1920x1080-noise_gauss-avif-cmyk-uchar-none-1.csv"), data_path("data12/1920x1080-noise_gauss-avif-cmyk-uchar-none-3.csv"), data_path("data12/1920x1080-noise_gauss-avif-cmyk-uchar-transparent-1.csv"), data_path("data12/1920x1080-noise_gauss-avif-cmyk-uchar-transparent-3.csv"), data_path("data12/1920x1080-noise_gauss-avif-srgb-float-none-1.csv"), data_path("data12/1920x1080-noise_gauss-avif-srgb-float-none-3.csv"), data_path("data12/1920x1080-noise_gauss-avif-srgb-float-transparent-1.csv"), data_path("data12/1920x1080-noise_gauss-avif-srgb-float-transparent-3.csv"), data_path("data12/1920x1080-noise_gauss-avif-srgb-uchar-none-1.csv"), data_path("data12/1920x1080-noise_gauss-avif-srgb-uchar-none-3.csv"), data_path("data12/1920x1080-noise_gauss-avif-srgb-uchar-transparent-1.csv"), data_path("data12/1920x1080-noise_gauss-avif-srgb-uchar-transparent-3.csv"), data_path("data12/1920x1080-noise_gauss-jpg-cmyk-float-none-1.csv"), data_path("data12/1920x1080-noise_gauss-jpg-cmyk-float-none-3.csv"), data_path("data12/1920x1080-noise_gauss-jpg-cmyk-float-transparent-1.csv"), data_path("data12/1920x1080-noise_gauss-jpg-cmyk-float-transparent-3.csv"), data_path("data12/1920x1080-noise_gauss-jpg-cmyk-uchar-none-1.csv"), data_path("data12/1920x1080-noise_gauss-jpg-cmyk-uchar-none-3.csv"), data_path("data12/1920x1080-noise_gauss-jpg-cmyk-uchar-transparent-1.csv"), data_path("data12/1920x1080-noise_gauss-jpg-cmyk-uchar-transparent-3.csv"), data_path("data12/1920x1080-noise_gauss-jpg-srgb-float-none-1.csv"), data_path("data12/1920x1080-noise_gauss-jpg-srgb-float-none-3.csv"), data_path("data12/1920x1080-noise_gauss-jpg-srgb-float-transparent-1.csv"), data_path("data12/1920x1080-noise_gauss-jpg-srgb-float-transparent-3.csv"), data_path("data12/1920x1080-noise_gauss-jpg-srgb-uchar-none-1.csv"), data_path("data12/1920x1080-noise_gauss-jpg-srgb-uchar-none-3.csv"), data_path("data12/1920x1080-noise_gauss-jpg-srgb-uchar-transparent-1.csv"), data_path("data12/1920x1080-noise_gauss-jpg-srgb-uchar-transparent-3.csv"), data_path("data12/1920x1080-solid_black-avif-cmyk-float-none-1.csv"), data_path("data12/1920x1080-solid_black-avif-cmyk-float-none-3.csv"), data_path("data12/1920x1080-solid_black-avif-cmyk-float-transparent-1.csv"), data_path("data12/1920x1080-solid_black-avif-cmyk-float-transparent-3.csv"), data_path("data12/1920x1080-solid_black-avif-cmyk-uchar-none-1.csv"), data_path("data12/1920x1080-solid_black-avif-cmyk-uchar-none-3.csv"), data_path("data12/1920x1080-solid_black-avif-cmyk-uchar-transparent-1.csv"), data_path("data12/1920x1080-solid_black-avif-cmyk-uchar-transparent-3.csv"), data_path("data12/1920x1080-solid_black-avif-srgb-float-none-1.csv"), data_path("data12/1920x1080-solid_black-avif-srgb-float-none-3.csv"), data_path("data12/1920x1080-solid_black-avif-srgb-float-transparent-1.csv"), data_path("data12/1920x1080-solid_black-avif-srgb-float-transparent-3.csv"), data_path("data12/1920x1080-solid_black-avif-srgb-uchar-none-1.csv"), data_path("data12/1920x1080-solid_black-avif-srgb-uchar-none-3.csv"), data_path("data12/1920x1080-solid_black-avif-srgb-uchar-transparent-1.csv"), data_path("data12/1920x1080-solid_black-avif-srgb-uchar-transparent-3.csv"), data_path("data12/1920x1080-solid_black-jpg-cmyk-float-none-1.csv"), data_path("data12/1920x1080-solid_black-jpg-cmyk-float-none-3.csv"), data_path("data12/1920x1080-solid_black-jpg-cmyk-float-transparent-1.csv"), data_path("data12/1920x1080-solid_black-jpg-cmyk-float-transparent-3.csv"), data_path("data12/1920x1080-solid_black-jpg-cmyk-uchar-none-1.csv"), data_path("data12/1920x1080-solid_black-jpg-cmyk-uchar-none-3.csv"), data_path("data12/1920x1080-solid_black-jpg-cmyk-uchar-transparent-1.csv"), data_path("data12/1920x1080-solid_black-jpg-cmyk-uchar-transparent-3.csv"), data_path("data12/1920x1080-solid_black-jpg-srgb-float-none-1.csv"), data_path("data12/1920x1080-solid_black-jpg-srgb-float-none-3.csv"), data_path("data12/1920x1080-solid_black-jpg-srgb-float-transparent-1.csv"), data_path("data12/1920x1080-solid_black-jpg-srgb-float-transparent-3.csv"), data_path("data12/1920x1080-solid_black-jpg-srgb-uchar-none-1.csv"), data_path("data12/1920x1080-solid_black-jpg-srgb-uchar-none-3.csv"), data_path("data12/1920x1080-solid_black-jpg-srgb-uchar-transparent-1.csv"), data_path("data12/1920x1080-solid_black-jpg-srgb-uchar-transparent-3.csv"), data_path("data12/1920x1080-texture_worley-avif-cmyk-float-none-1.csv"), data_path("data12/1920x1080-texture_worley-avif-cmyk-float-none-3.csv"), data_path("data12/1920x1080-texture_worley-avif-cmyk-float-transparent-1.csv"), data_path("data12/1920x1080-texture_worley-avif-cmyk-float-transparent-3.csv"), data_path("data12/1920x1080-texture_worley-avif-cmyk-uchar-none-1.csv"), data_path("data12/1920x1080-texture_worley-avif-cmyk-uchar-none-3.csv"), data_path("data12/1920x1080-texture_worley-avif-cmyk-uchar-transparent-1.csv"), data_path("data12/1920x1080-texture_worley-avif-cmyk-uchar-transparent-3.csv"), data_path("data12/1920x1080-texture_worley-avif-srgb-float-none-1.csv"), data_path("data12/1920x1080-texture_worley-avif-srgb-float-none-3.csv"), data_path("data12/1920x1080-texture_worley-avif-srgb-float-transparent-1.csv"), data_path("data12/1920x1080-texture_worley-avif-srgb-float-transparent-3.csv"), data_path("data12/1920x1080-texture_worley-avif-srgb-uchar-none-1.csv"), data_path("data12/1920x1080-texture_worley-avif-srgb-uchar-none-3.csv"), data_path("data12/1920x1080-texture_worley-avif-srgb-uchar-transparent-1.csv"), data_path("data12/1920x1080-texture_worley-avif-srgb-uchar-transparent-3.csv"), data_path("data12/1920x1080-texture_worley-jpg-cmyk-float-none-1.csv"), data_path("data12/1920x1080-texture_worley-jpg-cmyk-float-none-3.csv"), data_path("data12/1920x1080-texture_worley-jpg-cmyk-float-transparent-1.csv"), data_path("data12/1920x1080-texture_worley-jpg-cmyk-float-transparent-3.csv"), data_path("data12/1920x1080-texture_worley-jpg-cmyk-uchar-none-1.csv"), data_path("data12/1920x1080-texture_worley-jpg-cmyk-uchar-none-3.csv"), data_path("data12/1920x1080-texture_worley-jpg-cmyk-uchar-transparent-1.csv"), data_path("data12/1920x1080-texture_worley-jpg-cmyk-uchar-transparent-3.csv"), data_path("data12/1920x1080-texture_worley-jpg-srgb-float-none-1.csv"), data_path("data12/1920x1080-texture_worley-jpg-srgb-float-none-3.csv"), data_path("data12/1920x1080-texture_worley-jpg-srgb-float-transparent-1.csv"), data_path("data12/1920x1080-texture_worley-jpg-srgb-float-transparent-3.csv"), data_path("data12/1920x1080-texture_worley-jpg-srgb-uchar-none-1.csv"), data_path("data12/1920x1080-texture_worley-jpg-srgb-uchar-none-3.csv"), data_path("data12/1920x1080-texture_worley-jpg-srgb-uchar-transparent-1.csv"), data_path("data12/1920x1080-texture_worley-jpg-srgb-uchar-transparent-3.csv"), data_path("data12/2560x1440-noise_gauss-avif-cmyk-float-none-1.csv"), data_path("data12/2560x1440-noise_gauss-avif-cmyk-float-none-3.csv"), data_path("data12/2560x1440-noise_gauss-avif-cmyk-float-transparent-1.csv"), data_path("data12/2560x1440-noise_gauss-avif-cmyk-float-transparent-3.csv"), data_path("data12/2560x1440-noise_gauss-avif-cmyk-uchar-none-1.csv"), data_path("data12/2560x1440-noise_gauss-avif-cmyk-uchar-none-3.csv"), data_path("data12/2560x1440-noise_gauss-avif-cmyk-uchar-transparent-1.csv"), data_path("data12/2560x1440-noise_gauss-avif-cmyk-uchar-transparent-3.csv"), data_path("data12/2560x1440-noise_gauss-avif-srgb-float-none-1.csv"), data_path("data12/2560x1440-noise_gauss-avif-srgb-float-none-3.csv"), data_path("data12/2560x1440-noise_gauss-avif-srgb-float-transparent-1.csv"), data_path("data12/2560x1440-noise_gauss-avif-srgb-float-transparent-3.csv"), data_path("data12/2560x1440-noise_gauss-avif-srgb-uchar-none-1.csv"), data_path("data12/2560x1440-noise_gauss-avif-srgb-uchar-none-3.csv"), data_path("data12/2560x1440-noise_gauss-avif-srgb-uchar-transparent-1.csv"), data_path("data12/2560x1440-noise_gauss-avif-srgb-uchar-transparent-3.csv"), data_path("data12/2560x1440-noise_gauss-jpg-cmyk-float-none-1.csv"), data_path("data12/2560x1440-noise_gauss-jpg-cmyk-float-none-3.csv"), data_path("data12/2560x1440-noise_gauss-jpg-cmyk-float-transparent-1.csv"), data_path("data12/2560x1440-noise_gauss-jpg-cmyk-float-transparent-3.csv"), data_path("data12/2560x1440-noise_gauss-jpg-cmyk-uchar-none-1.csv"), data_path("data12/2560x1440-noise_gauss-jpg-cmyk-uchar-none-3.csv"), data_path("data12/2560x1440-noise_gauss-jpg-cmyk-uchar-transparent-1.csv"), data_path("data12/2560x1440-noise_gauss-jpg-cmyk-uchar-transparent-3.csv"), data_path("data12/2560x1440-noise_gauss-jpg-srgb-float-none-1.csv"), data_path("data12/2560x1440-noise_gauss-jpg-srgb-float-none-3.csv"), data_path("data12/2560x1440-noise_gauss-jpg-srgb-float-transparent-1.csv"), data_path("data12/2560x1440-noise_gauss-jpg-srgb-float-transparent-3.csv"), data_path("data12/2560x1440-noise_gauss-jpg-srgb-uchar-none-1.csv"), data_path("data12/2560x1440-noise_gauss-jpg-srgb-uchar-none-3.csv"), data_path("data12/2560x1440-noise_gauss-jpg-srgb-uchar-transparent-1.csv"), data_path("data12/2560x1440-noise_gauss-jpg-srgb-uchar-transparent-3.csv"), data_path("data12/2560x1440-solid_black-avif-cmyk-float-none-1.csv"), data_path("data12/2560x1440-solid_black-avif-cmyk-float-none-3.csv"), data_path("data12/2560x1440-solid_black-avif-cmyk-float-transparent-1.csv"), data_path("data12/2560x1440-solid_black-avif-cmyk-float-transparent-3.csv"), data_path("data12/2560x1440-solid_black-avif-cmyk-uchar-none-1.csv"), data_path("data12/2560x1440-solid_black-avif-cmyk-uchar-none-3.csv"), data_path("data12/2560x1440-solid_black-avif-cmyk-uchar-transparent-1.csv"), data_path("data12/2560x1440-solid_black-avif-cmyk-uchar-transparent-3.csv"), data_path("data12/2560x1440-solid_black-avif-srgb-float-none-1.csv"), data_path("data12/2560x1440-solid_black-avif-srgb-float-none-3.csv"), data_path("data12/2560x1440-solid_black-avif-srgb-float-transparent-1.csv"), data_path("data12/2560x1440-solid_black-avif-srgb-float-transparent-3.csv"), data_path("data12/2560x1440-solid_black-avif-srgb-uchar-none-1.csv"), data_path("data12/2560x1440-solid_black-avif-srgb-uchar-none-3.csv"), data_path("data12/2560x1440-solid_black-avif-srgb-uchar-transparent-1.csv"), data_path("data12/2560x1440-solid_black-avif-srgb-uchar-transparent-3.csv"), data_path("data12/2560x1440-solid_black-jpg-cmyk-float-none-1.csv"), data_path("data12/2560x1440-solid_black-jpg-cmyk-float-none-3.csv"), data_path("data12/2560x1440-solid_black-jpg-cmyk-float-transparent-1.csv"), data_path("data12/2560x1440-solid_black-jpg-cmyk-float-transparent-3.csv"), data_path("data12/2560x1440-solid_black-jpg-cmyk-uchar-none-1.csv"), data_path("data12/2560x1440-solid_black-jpg-cmyk-uchar-none-3.csv"), data_path("data12/2560x1440-solid_black-jpg-cmyk-uchar-transparent-1.csv"), data_path("data12/2560x1440-solid_black-jpg-cmyk-uchar-transparent-3.csv"), data_path("data12/2560x1440-solid_black-jpg-srgb-float-none-1.csv"), data_path("data12/2560x1440-solid_black-jpg-srgb-float-none-3.csv"), data_path("data12/2560x1440-solid_black-jpg-srgb-float-transparent-1.csv"), data_path("data12/2560x1440-solid_black-jpg-srgb-float-transparent-3.csv"), data_path("data12/2560x1440-solid_black-jpg-srgb-uchar-none-1.csv"), data_path("data12/2560x1440-solid_black-jpg-srgb-uchar-none-3.csv"), data_path("data12/2560x1440-solid_black-jpg-srgb-uchar-transparent-1.csv"), data_path("data12/2560x1440-solid_black-jpg-srgb-uchar-transparent-3.csv"), data_path("data12/2560x1440-texture_worley-avif-cmyk-float-none-1.csv"), data_path("data12/2560x1440-texture_worley-avif-cmyk-float-none-3.csv"), data_path("data12/2560x1440-texture_worley-avif-cmyk-float-transparent-1.csv"), data_path("data12/2560x1440-texture_worley-avif-cmyk-float-transparent-3.csv"), data_path("data12/2560x1440-texture_worley-avif-cmyk-uchar-none-1.csv"), data_path("data12/2560x1440-texture_worley-avif-cmyk-uchar-none-3.csv"), data_path("data12/2560x1440-texture_worley-avif-cmyk-uchar-transparent-1.csv"), data_path("data12/2560x1440-texture_worley-avif-cmyk-uchar-transparent-3.csv"), data_path("data12/2560x1440-texture_worley-avif-srgb-float-none-1.csv"), data_path("data12/2560x1440-texture_worley-avif-srgb-float-none-3.csv"), data_path("data12/2560x1440-texture_worley-avif-srgb-float-transparent-1.csv"), data_path("data12/2560x1440-texture_worley-avif-srgb-float-transparent-3.csv"), data_path("data12/2560x1440-texture_worley-avif-srgb-uchar-none-1.csv"), data_path("data12/2560x1440-texture_worley-avif-srgb-uchar-none-3.csv"), data_path("data12/2560x1440-texture_worley-avif-srgb-uchar-transparent-1.csv"), data_path("data12/2560x1440-texture_worley-avif-srgb-uchar-transparent-3.csv"), data_path("data12/2560x1440-texture_worley-jpg-cmyk-float-none-1.csv"), data_path("data12/2560x1440-texture_worley-jpg-cmyk-float-none-3.csv"), data_path("data12/2560x1440-texture_worley-jpg-cmyk-float-transparent-1.csv"), data_path("data12/2560x1440-texture_worley-jpg-cmyk-float-transparent-3.csv"), data_path("data12/2560x1440-texture_worley-jpg-cmyk-uchar-none-1.csv"), data_path("data12/2560x1440-texture_worley-jpg-cmyk-uchar-none-3.csv"), data_path("data12/2560x1440-texture_worley-jpg-cmyk-uchar-transparent-1.csv"), data_path("data12/2560x1440-texture_worley-jpg-cmyk-uchar-transparent-3.csv"), data_path("data12/2560x1440-texture_worley-jpg-srgb-float-none-1.csv"), data_path("data12/2560x1440-texture_worley-jpg-srgb-float-none-3.csv"), data_path("data12/2560x1440-texture_worley-jpg-srgb-float-transparent-1.csv"), data_path("data12/2560x1440-texture_worley-jpg-srgb-float-transparent-3.csv"), data_path("data12/2560x1440-texture_worley-jpg-srgb-uchar-none-1.csv"), data_path("data12/2560x1440-texture_worley-jpg-srgb-uchar-none-3.csv"), data_path("data12/2560x1440-texture_worley-jpg-srgb-uchar-transparent-1.csv"), data_path("data12/2560x1440-texture_worley-jpg-srgb-uchar-transparent-3.csv"), data_path("data12/3840x2160-noise_gauss-avif-cmyk-float-none-1.csv"), data_path("data12/3840x2160-noise_gauss-avif-cmyk-float-none-3.csv"), data_path("data12/3840x2160-noise_gauss-avif-cmyk-float-transparent-1.csv"), data_path("data12/3840x2160-noise_gauss-avif-cmyk-float-transparent-3.csv"), data_path("data12/3840x2160-noise_gauss-avif-cmyk-uchar-none-1.csv"), data_path("data12/3840x2160-noise_gauss-avif-cmyk-uchar-none-3.csv"), data_path("data12/3840x2160-noise_gauss-avif-cmyk-uchar-transparent-1.csv"), data_path("data12/3840x2160-noise_gauss-avif-cmyk-uchar-transparent-3.csv"), data_path("data12/3840x2160-noise_gauss-avif-srgb-float-none-1.csv"), data_path("data12/3840x2160-noise_gauss-avif-srgb-float-none-3.csv"), data_path("data12/3840x2160-noise_gauss-avif-srgb-float-transparent-1.csv"), data_path("data12/3840x2160-noise_gauss-avif-srgb-float-transparent-3.csv"), data_path("data12/3840x2160-noise_gauss-avif-srgb-uchar-none-1.csv"), data_path("data12/3840x2160-noise_gauss-avif-srgb-uchar-none-3.csv"), data_path("data12/3840x2160-noise_gauss-avif-srgb-uchar-transparent-1.csv"), data_path("data12/3840x2160-noise_gauss-avif-srgb-uchar-transparent-3.csv"), data_path("data12/3840x2160-noise_gauss-jpg-cmyk-float-none-1.csv"), data_path("data12/3840x2160-noise_gauss-jpg-cmyk-float-none-3.csv"), data_path("data12/3840x2160-noise_gauss-jpg-cmyk-float-transparent-1.csv"), data_path("data12/3840x2160-noise_gauss-jpg-cmyk-float-transparent-3.csv"), data_path("data12/3840x2160-noise_gauss-jpg-cmyk-uchar-none-1.csv"), data_path("data12/3840x2160-noise_gauss-jpg-cmyk-uchar-none-3.csv"), data_path("data12/3840x2160-noise_gauss-jpg-cmyk-uchar-transparent-1.csv"), data_path("data12/3840x2160-noise_gauss-jpg-cmyk-uchar-transparent-3.csv"), data_path("data12/3840x2160-noise_gauss-jpg-srgb-float-none-1.csv"), data_path("data12/3840x2160-noise_gauss-jpg-srgb-float-none-3.csv"), data_path("data12/3840x2160-noise_gauss-jpg-srgb-float-transparent-1.csv"), data_path("data12/3840x2160-noise_gauss-jpg-srgb-float-transparent-3.csv"), data_path("data12/3840x2160-noise_gauss-jpg-srgb-uchar-none-1.csv"), data_path("data12/3840x2160-noise_gauss-jpg-srgb-uchar-none-3.csv"), data_path("data12/3840x2160-noise_gauss-jpg-srgb-uchar-transparent-1.csv"), data_path("data12/3840x2160-noise_gauss-jpg-srgb-uchar-transparent-3.csv"), data_path("data12/3840x2160-solid_black-avif-cmyk-float-none-1.csv"), data_path("data12/3840x2160-solid_black-avif-cmyk-float-none-3.csv"), data_path("data12/3840x2160-solid_black-avif-cmyk-float-transparent-1.csv"), data_path("data12/3840x2160-solid_black-avif-cmyk-float-transparent-3.csv"), data_path("data12/3840x2160-solid_black-avif-cmyk-uchar-none-1.csv"), data_path("data12/3840x2160-solid_black-avif-cmyk-uchar-none-3.csv"), data_path("data12/3840x2160-solid_black-avif-cmyk-uchar-transparent-1.csv"), data_path("data12/3840x2160-solid_black-avif-cmyk-uchar-transparent-3.csv"), data_path("data12/3840x2160-solid_black-avif-srgb-float-none-1.csv"), data_path("data12/3840x2160-solid_black-avif-srgb-float-none-3.csv"), data_path("data12/3840x2160-solid_black-avif-srgb-float-transparent-1.csv"), data_path("data12/3840x2160-solid_black-avif-srgb-float-transparent-3.csv"), data_path("data12/3840x2160-solid_black-avif-srgb-uchar-none-1.csv"), data_path("data12/3840x2160-solid_black-avif-srgb-uchar-none-3.csv"), data_path("data12/3840x2160-solid_black-avif-srgb-uchar-transparent-1.csv"), data_path("data12/3840x2160-solid_black-avif-srgb-uchar-transparent-3.csv"), data_path("data12/3840x2160-solid_black-jpg-cmyk-float-none-1.csv"), data_path("data12/3840x2160-solid_black-jpg-cmyk-float-none-3.csv"), data_path("data12/3840x2160-solid_black-jpg-cmyk-float-transparent-1.csv"), data_path("data12/3840x2160-solid_black-jpg-cmyk-float-transparent-3.csv"), data_path("data12/3840x2160-solid_black-jpg-cmyk-uchar-none-1.csv"), data_path("data12/3840x2160-solid_black-jpg-cmyk-uchar-none-3.csv"), data_path("data12/3840x2160-solid_black-jpg-cmyk-uchar-transparent-1.csv"), data_path("data12/3840x2160-solid_black-jpg-cmyk-uchar-transparent-3.csv"), data_path("data12/3840x2160-solid_black-jpg-srgb-float-none-1.csv"), data_path("data12/3840x2160-solid_black-jpg-srgb-float-none-3.csv"), data_path("data12/3840x2160-solid_black-jpg-srgb-float-transparent-1.csv"), data_path("data12/3840x2160-solid_black-jpg-srgb-float-transparent-3.csv"), data_path("data12/3840x2160-solid_black-jpg-srgb-uchar-none-1.csv"), data_path("data12/3840x2160-solid_black-jpg-srgb-uchar-none-3.csv"), data_path("data12/3840x2160-solid_black-jpg-srgb-uchar-transparent-1.csv"), data_path("data12/3840x2160-solid_black-jpg-srgb-uchar-transparent-3.csv"), data_path("data12/3840x2160-texture_worley-avif-cmyk-float-none-1.csv"), data_path("data12/3840x2160-texture_worley-avif-cmyk-float-none-3.csv"), data_path("data12/3840x2160-texture_worley-avif-cmyk-float-transparent-1.csv"), data_path("data12/3840x2160-texture_worley-avif-cmyk-float-transparent-3.csv"), data_path("data12/3840x2160-texture_worley-avif-cmyk-uchar-none-1.csv"), data_path("data12/3840x2160-texture_worley-avif-cmyk-uchar-none-3.csv"), data_path("data12/3840x2160-texture_worley-avif-cmyk-uchar-transparent-1.csv"), data_path("data12/3840x2160-texture_worley-avif-cmyk-uchar-transparent-3.csv"), data_path("data12/3840x2160-texture_worley-avif-srgb-float-none-1.csv"), data_path("data12/3840x2160-texture_worley-avif-srgb-float-none-3.csv"), data_path("data12/3840x2160-texture_worley-avif-srgb-float-transparent-1.csv"), data_path("data12/3840x2160-texture_worley-avif-srgb-float-transparent-3.csv"), data_path("data12/3840x2160-texture_worley-avif-srgb-uchar-none-1.csv"), data_path("data12/3840x2160-texture_worley-avif-srgb-uchar-none-3.csv"), data_path("data12/3840x2160-texture_worley-avif-srgb-uchar-transparent-1.csv"), data_path("data12/3840x2160-texture_worley-avif-srgb-uchar-transparent-3.csv"), data_path("data12/3840x2160-texture_worley-jpg-cmyk-float-none-1.csv"), data_path("data12/3840x2160-texture_worley-jpg-cmyk-float-none-3.csv"), data_path("data12/3840x2160-texture_worley-jpg-cmyk-float-transparent-1.csv"), data_path("data12/3840x2160-texture_worley-jpg-cmyk-float-transparent-3.csv"), data_path("data12/3840x2160-texture_worley-jpg-cmyk-uchar-none-1.csv"), data_path("data12/3840x2160-texture_worley-jpg-cmyk-uchar-none-3.csv"), data_path("data12/3840x2160-texture_worley-jpg-cmyk-uchar-transparent-1.csv"), data_path("data12/3840x2160-texture_worley-jpg-cmyk-uchar-transparent-3.csv"), data_path("data12/3840x2160-texture_worley-jpg-srgb-float-none-1.csv"), data_path("data12/3840x2160-texture_worley-jpg-srgb-float-none-3.csv"), data_path("data12/3840x2160-texture_worley-jpg-srgb-float-transparent-1.csv"), data_path("data12/3840x2160-texture_worley-jpg-srgb-float-transparent-3.csv"), data_path("data12/3840x2160-texture_worley-jpg-srgb-uchar-none-1.csv"), data_path("data12/3840x2160-texture_worley-jpg-srgb-uchar-none-3.csv"), data_path("data12/3840x2160-texture_worley-jpg-srgb-uchar-transparent-1.csv"), data_path("data12/3840x2160-texture_worley-jpg-srgb-uchar-transparent-3.csv")]
env_list13 = [data_path("data13/1-1-1024-10.csv"), data_path("data13/1-1-1024-120.csv"), data_path("data13/1-1-1024-180.csv"), data_path("data13/1-1-1024-60.csv"), data_path("data13/1-1-16384-10.csv"), data_path("data13/1-1-16384-120.csv"), data_path("data13/1-1-16384-180.csv"), data_path("data13/1-1-16384-60.csv"), data_path("data13/1-1-4096-10.csv"), data_path("data13/1-1-4096-120.csv"), data_path("data13/1-1-4096-180.csv"), data_path("data13/1-1-4096-60.csv"), data_path("data13/1-10-1024-10.csv"), data_path("data13/1-10-1024-120.csv"), data_path("data13/1-10-1024-180.csv"), data_path("data13/1-10-1024-60.csv"), data_path("data13/1-10-16384-10.csv"), data_path("data13/1-10-16384-120.csv"), data_path("data13/1-10-16384-180.csv"), data_path("data13/1-10-16384-60.csv"), data_path("data13/1-10-4096-10.csv"), data_path("data13/1-10-4096-120.csv"), data_path("data13/1-10-4096-180.csv"), data_path("data13/1-10-4096-60.csv"), data_path("data13/1-100-1024-10.csv"), data_path("data13/1-100-1024-120.csv"), data_path("data13/1-100-1024-180.csv"), data_path("data13/1-100-1024-60.csv"), data_path("data13/1-100-16384-10.csv"), data_path("data13/1-100-16384-120.csv"), data_path("data13/1-100-16384-180.csv"), data_path("data13/1-100-16384-60.csv"), data_path("data13/1-100-4096-10.csv"), data_path("data13/1-100-4096-120.csv"), data_path("data13/1-100-4096-180.csv"), data_path("data13/1-100-4096-60.csv"), data_path("data13/10-1-1024-10.csv"), data_path("data13/10-1-1024-120.csv"), data_path("data13/10-1-1024-180.csv"), data_path("data13/10-1-1024-60.csv"), data_path("data13/10-1-16384-10.csv"), data_path("data13/10-1-16384-120.csv"), data_path("data13/10-1-16384-180.csv"), data_path("data13/10-1-16384-60.csv"), data_path("data13/10-1-4096-10.csv"), data_path("data13/10-1-4096-120.csv"), data_path("data13/10-1-4096-180.csv"), data_path("data13/10-1-4096-60.csv"), data_path("data13/10-10-1024-10.csv"), data_path("data13/10-10-1024-120.csv"), data_path("data13/10-10-1024-180.csv"), data_path("data13/10-10-1024-60.csv"), data_path("data13/10-10-16384-10.csv"), data_path("data13/10-10-16384-120.csv"), data_path("data13/10-10-16384-180.csv"), data_path("data13/10-10-16384-60.csv"), data_path("data13/10-10-4096-10.csv"), data_path("data13/10-10-4096-120.csv"), data_path("data13/10-10-4096-180.csv"), data_path("data13/10-10-4096-60.csv"), data_path("data13/10-100-1024-10.csv"), data_path("data13/10-100-1024-120.csv"), data_path("data13/10-100-1024-180.csv"), data_path("data13/10-100-1024-60.csv"), data_path("data13/10-100-16384-10.csv"), data_path("data13/10-100-16384-120.csv"), data_path("data13/10-100-16384-180.csv"), data_path("data13/10-100-16384-60.csv"), data_path("data13/10-100-4096-10.csv"), data_path("data13/10-100-4096-120.csv"), data_path("data13/10-100-4096-180.csv"), data_path("data13/10-100-4096-60.csv"), data_path("data13/50-1-1024-10.csv"), data_path("data13/50-1-1024-120.csv"), data_path("data13/50-1-1024-180.csv"), data_path("data13/50-1-1024-60.csv"), data_path("data13/50-1-16384-10.csv"), data_path("data13/50-1-16384-120.csv"), data_path("data13/50-1-16384-180.csv"), data_path("data13/50-1-16384-60.csv"), data_path("data13/50-1-4096-10.csv"), data_path("data13/50-1-4096-120.csv"), data_path("data13/50-1-4096-180.csv"), data_path("data13/50-1-4096-60.csv"), data_path("data13/50-10-1024-10.csv"), data_path("data13/50-10-1024-120.csv"), data_path("data13/50-10-1024-180.csv"), data_path("data13/50-10-1024-60.csv"), data_path("data13/50-10-16384-10.csv"), data_path("data13/50-10-16384-120.csv"), data_path("data13/50-10-16384-180.csv"), data_path("data13/50-10-16384-60.csv"), data_path("data13/50-10-4096-10.csv"), data_path("data13/50-10-4096-120.csv"), data_path("data13/50-10-4096-180.csv"), data_path("data13/50-10-4096-60.csv"), data_path("data13/50-100-1024-10.csv"), data_path("data13/50-100-1024-120.csv"), data_path("data13/50-100-1024-180.csv"), data_path("data13/50-100-1024-60.csv"), data_path("data13/50-100-16384-10.csv"), data_path("data13/50-100-16384-120.csv"), data_path("data13/50-100-16384-180.csv"), data_path("data13/50-100-16384-60.csv"), data_path("data13/50-100-4096-10.csv"), data_path("data13/50-100-4096-120.csv"), data_path("data13/50-100-4096-180.csv"), data_path("data13/50-100-4096-60.csv")]
env_list14 = [data_path("data14/1-1-1024-10.csv"), data_path("data14/1-1-1024-120.csv"), data_path("data14/1-1-1024-180.csv"), data_path("data14/1-1-1024-60.csv"), data_path("data14/1-1-16384-10.csv"), data_path("data14/1-1-16384-120.csv"), data_path("data14/1-1-16384-180.csv"), data_path("data14/1-1-16384-60.csv"), data_path("data14/1-1-4096-10.csv"), data_path("data14/1-1-4096-120.csv"), data_path("data14/1-1-4096-180.csv"), data_path("data14/1-1-4096-60.csv"), data_path("data14/1-10-1024-10.csv"), data_path("data14/1-10-1024-120.csv"), data_path("data14/1-10-1024-180.csv"), data_path("data14/1-10-1024-60.csv"), data_path("data14/1-10-16384-10.csv"), data_path("data14/1-10-16384-120.csv"), data_path("data14/1-10-16384-180.csv"), data_path("data14/1-10-16384-60.csv"), data_path("data14/1-10-4096-10.csv"), data_path("data14/1-10-4096-120.csv"), data_path("data14/1-10-4096-180.csv"), data_path("data14/1-10-4096-60.csv"), data_path("data14/1-100-1024-10.csv"), data_path("data14/1-100-1024-120.csv"), data_path("data14/1-100-1024-180.csv"), data_path("data14/1-100-1024-60.csv"), data_path("data14/1-100-16384-10.csv"), data_path("data14/1-100-16384-120.csv"), data_path("data14/1-100-16384-180.csv"), data_path("data14/1-100-16384-60.csv"), data_path("data14/1-100-4096-10.csv"), data_path("data14/1-100-4096-120.csv"), data_path("data14/1-100-4096-180.csv"), data_path("data14/1-100-4096-60.csv"), data_path("data14/10-1-1024-10.csv"), data_path("data14/10-1-1024-120.csv"), data_path("data14/10-1-1024-180.csv"), data_path("data14/10-1-1024-60.csv"), data_path("data14/10-1-16384-10.csv"), data_path("data14/10-1-16384-120.csv"), data_path("data14/10-1-16384-180.csv"), data_path("data14/10-1-16384-60.csv"), data_path("data14/10-1-4096-10.csv"), data_path("data14/10-1-4096-120.csv"), data_path("data14/10-1-4096-180.csv"), data_path("data14/10-1-4096-60.csv"), data_path("data14/10-10-1024-10.csv"), data_path("data14/10-10-1024-120.csv"), data_path("data14/10-10-1024-180.csv"), data_path("data14/10-10-1024-60.csv"), data_path("data14/10-10-16384-10.csv"), data_path("data14/10-10-16384-120.csv"), data_path("data14/10-10-16384-180.csv"), data_path("data14/10-10-16384-60.csv"), data_path("data14/10-10-4096-10.csv"), data_path("data14/10-10-4096-120.csv"), data_path("data14/10-10-4096-180.csv"), data_path("data14/10-10-4096-60.csv"), data_path("data14/10-100-1024-10.csv"), data_path("data14/10-100-1024-120.csv"), data_path("data14/10-100-1024-180.csv"), data_path("data14/10-100-1024-60.csv"), data_path("data14/10-100-16384-10.csv"), data_path("data14/10-100-16384-120.csv"), data_path("data14/10-100-16384-180.csv"), data_path("data14/10-100-16384-60.csv"), data_path("data14/10-100-4096-10.csv"), data_path("data14/10-100-4096-120.csv"), data_path("data14/10-100-4096-180.csv"), data_path("data14/10-100-4096-60.csv"), data_path("data14/50-1-1024-10.csv"), data_path("data14/50-1-1024-120.csv"), data_path("data14/50-1-1024-180.csv"), data_path("data14/50-1-1024-60.csv"), data_path("data14/50-1-16384-10.csv"), data_path("data14/50-1-16384-120.csv"), data_path("data14/50-1-16384-180.csv"), data_path("data14/50-1-16384-60.csv"), data_path("data14/50-1-4096-10.csv"), data_path("data14/50-1-4096-120.csv"), data_path("data14/50-1-4096-180.csv"), data_path("data14/50-1-4096-60.csv"), data_path("data14/50-10-1024-10.csv"), data_path("data14/50-10-1024-120.csv"), data_path("data14/50-10-1024-180.csv"), data_path("data14/50-10-1024-60.csv"), data_path("data14/50-10-16384-10.csv"), data_path("data14/50-10-16384-120.csv"), data_path("data14/50-10-16384-180.csv"), data_path("data14/50-10-16384-60.csv"), data_path("data14/50-10-4096-10.csv"), data_path("data14/50-10-4096-120.csv"), data_path("data14/50-10-4096-180.csv"), data_path("data14/50-10-4096-60.csv"), data_path("data14/50-100-1024-10.csv"), data_path("data14/50-100-1024-120.csv"), data_path("data14/50-100-1024-180.csv"), data_path("data14/50-100-1024-60.csv"), data_path("data14/50-100-16384-10.csv"), data_path("data14/50-100-16384-120.csv"), data_path("data14/50-100-16384-180.csv"), data_path("data14/50-100-16384-60.csv"), data_path("data14/50-100-4096-10.csv"), data_path("data14/50-100-4096-120.csv"), data_path("data14/50-100-4096-180.csv"), data_path("data14/50-100-4096-60.csv")]





def get_whole_data(path):
    df = pd.read_csv(path)
    ndarray = df.values
    return ndarray


def get_attributes_result(data):
    X = data[:, :-1]
    Y = data[:, -1][:, np.newaxis]
    return X, Y


def generate_unique_random_numbers(n, seed=0):
    # 创建一个包含0到n的数组
    # random.seed(seed)
    numbers = list(range(n + 1))
    # Fisher-Yates shuffle算法
    for i in range(len(numbers) - 1, 0, -1):
        j = random.randint(0, i)
        numbers[i], numbers[j] = numbers[j], numbers[i]

    return numbers

system_sample_rate = {
    "lighttpd": (0.03, 0.06),
    "spear": (0.1, 0.15)
}
def construct_drift_stream(
    system_name,
    envs,
    n_env,
    recurring,
    minr=0.4,
    maxr=0.8,
    max_count=1500,
    min_count=400
):
    """
    Construct a data stream with concept drift.

    Parameters
    ----------
    system_name : str
        Name of the software system.

    envs : list[str]
        List of csv file paths, each representing an environment.

    n_env : int
        Number of environments (drifts) in the stream.

    recurring : bool
        Whether recurring environments exist.

    minr : float
        Minimum sampling ratio.

    maxr : float
        Maximum sampling ratio.

    max_count : int
        Maximum number of samples for one environment.

    min_count : int
        Minimum number of samples for one environment.

    Returns
    -------
    dict
        {
            "system": system_name,
            "init_data": DataFrame,
            "whole_data": list,
            "sequence": list
        }
    """

    if system_name in system_sample_rate:
        minr, maxr = system_sample_rate[system_name]

    n_drift = n_env

    # ---------- determine environment sequence ----------
    if recurring:

        # select a subset of environments
        k = max(1, n_drift // 2)
        k = min(k, len(envs))

        env_subset = random.sample(envs, k)

        env_sequence = []

        # first environment
        prev_env = random.choice(env_subset)
        env_sequence.append(prev_env)

        # remaining environments
        for _ in range(n_drift - 1):

            # avoid consecutive duplicate environments
            candidates = [e for e in env_subset if e != prev_env]

            if len(candidates) == 0:
                next_env = prev_env
            else:
                next_env = random.choice(candidates)

            env_sequence.append(next_env)
            prev_env = next_env

    else:

        if n_drift > len(envs):
            n_drift = len(envs)

        env_sequence = random.sample(envs, n_drift)

    init_data = []
    test_data = []

    print(f"env sequence: {env_sequence}")

    # ---------- build stream ----------
    for i, env_path in enumerate(env_sequence):

        df = pd.read_csv(env_path)

        min_n_samples = int(minr * len(df))
        max_n_samples = int(maxr * len(df))

        # apply sample count constraints
        max_n_samples = min(max_n_samples, max_count)
        min_n_samples = max(min_n_samples, min_count)

        # ensure valid range
        if max_n_samples < min_n_samples:
            max_n_samples = max_count
            min_n_samples = min_count

        sample_size = random.randint(min_n_samples, max_n_samples)
        sample_size = max(1, sample_size)

        # avoid sampling more than dataset size
        sample_size = min(sample_size, len(df))

        sampled_df = df.sample(n=sample_size, replace=False)

        if i == 0:
            # first environment used for initialization

            if len(sampled_df) <= 50:
                init_part = sampled_df
                test_part = pd.DataFrame(columns=sampled_df.columns)

            else:
                init_part = sampled_df.sample(n=50)
                test_part = sampled_df.drop(init_part.index)

            init_data = init_part.values
            test_data.append(test_part.values)

        else:
            test_data.append(sampled_df.values)

    return {
        "system": system_name,
        "init_data": init_data,
        "whole_data": test_data,
        "sequence": env_sequence
    }


def process_training_data(multi_env_list=[], n_drifts=4, min_occupation=0.4, max_occupation=0.8, max_count=2000, min_count=200,
                          init_train_count=50, reuse_test=False, given_list=None):
    if len(multi_env_list)>30:
        max_count = 800
    count_list = {}
    env_count = len(multi_env_list)
    selected_list = generate_unique_random_numbers(env_count - 1)
    whole_data = None
    change_time = 0
    init_data = None
    count = 0
    change_point = list()
    if not reuse_test:
        random_list = selected_list
    elif given_list is not None:
        random_list = given_list
    else:
        selected_list = selected_list[:6]
        random_list = selected_list * 2
    logging.info(f"data list {random_list}")
    """
    处理逻辑：
        如果reuse_test False && given_list = None ,则随机选择n_drifts个数据集,每个数据集随机采样一部分数据,作为流数据;
        如果reuse_test True && given_list != None,则按照given_list的顺序选择数据集,每个数据集随机采样一部分数据,作为流数据;
        如果reuse_test True && given_list = None, 则随机选择3个数据集,每个数据集随机采样一部分数据,作为流数据;然后这3个数据集重复一次,作为流数据;
    """
    for i in random_list:
        if count == n_drifts:
            break

        whole_data_temp = get_whole_data(multi_env_list[i])
        data_num = len(whole_data_temp)
        low = int(data_num * min_occupation)
        high = int(data_num * max_occupation)
        sample_count = random.randint(low, high)
        if sample_count > max_count:
            sample_count = max_count
        dataset_name = multi_env_list[i].split('\\')[-1]
        count_list[count] = [dataset_name, sample_count]
        count += 1

        all_index = np.array(list(range(data_num)))
        # np.random.seed(seed)
        data_index = np.random.choice(all_index, size=sample_count)
        data = whole_data_temp[data_index]
        change_time += 1

        if whole_data is None:
            # whole_data = data
            whole_data = []
            init_count = len(data)
            init_test_count = init_count - init_train_count
            all_index_temp = np.array(list(range(len(data))))

            train_index_temp = np.random.choice(all_index_temp, size=init_train_count)
            train_data = data[train_index_temp]
            test_index_temp = np.setdiff1d(all_index_temp, train_index_temp)
            test_data = data[test_index_temp]
            whole_data.append(test_data)
            init_data = train_data
        else:
            whole_data.append(data)

        change_point.append(len(whole_data))
    # print(f"change time is {change_time}")
    logging.info(f"change time is {change_time}")
    print("data info")
    # print(f"init train data count = {init_train_count}, stream data count = {len(whole_data)}")
    # logging.info(f"init train data count = {init_train_count}, stream data count = {len(whole_data)}")
    true_change_point = []
    all = 0
    for env_data in whole_data:
        all += len(env_data)
        true_change_point.append(len(env_data))
    data_count_list = []
    for data in whole_data:
        data_count_list.append(len(data))
    logging.info(f"test env list: {random_list}, test env count: {data_count_list}")
    return {'init_data': init_data, 'whole_data': whole_data, 'true_change_point': true_change_point}


def plot_show(performance_dict, save_fig, save_path=None, title=None):
    plt.figure()
    mean_perf = {}
    for k, v in performance_dict.items():
        length = len(v)
        x = list(range(length))
        plt.plot(x, v, label=k)
        mean_perf[k] = round(np.mean(v), 4)
    plt.xlabel(mean_perf)
    plt.legend()
    plt.title(title)
    if save_fig:
        plt.savefig(save_path)
        # logging.info(f"figure is saved, path={file_path}")
    # plt.show()

def test_dhda_reuse(name, test_data):
    data = construct_drift_stream(name, test_data, 6, recurring=True)
    test_data_list = data['whole_data']
    train_data = data['init_data']
    train_x, train_y = get_attributes_result(train_data)

    """
    An ideal model refers to a model that 
        1. knows when concept drift occurs, 
        2. knows what concept it will transform into, and 
        3. can accurately divide history and reuse historical models.
    An enhance model refers to a model that
        1. don't knows when concept drift occurs, but depends on detector
        2. knows what concept it will transform into, and
        3. can accurately divide history and reuse historical models. 
    An normal model refers to a model that
        1. don't knows when concept drift occurs, but depends on detector
        2. don't knows what concept it will transform into, and
        3. reactive to handle concept drift.
    """
    performance_dict = {}
    time_cost_dict = {}

    logging.info("--------dhda reuse model with all--------")
    random.seed(seed)
    np.random.seed(seed)
    dhda_reuse_ALL_model = DaL_Regressor(MODEL_REUSE=True, Smooth=True, CPD=True, min_sample=3)
    # dhda_model.fit_model(train_x, np.ravel(train_y))
    dhda_reuse_ALL_model.fit_model(train_x, train_y)

    dhda_reuse_ALL_error_list = []
    dhda_reuse_ALL_y = []

    c = 0
    time0 = time.time()
    for index, env_data in enumerate(test_data_list):
        logging.info(f"test env: {index}, timestep: {c}")
        env_test_x, env_test_y = get_attributes_result(env_data)
        test_size = len(env_test_y)

        for i in range(test_size):
            c += 1
            y_pre = dhda_reuse_ALL_model.predict(env_test_x[i][np.newaxis, :])
            val = 0
            # val = abs(y_pre - y_test[i])[0][0] change
            if env_test_y[i] != 0:
                val = ((abs(y_pre - env_test_y[i])[0]) / env_test_y[i])[0]
                dhda_reuse_ALL_error_list.append(val)
                dhda_reuse_ALL_model.learn_data(env_test_x[i][np.newaxis, :], env_test_y[i][np.newaxis, :], val)
            if c % 32 == 0:
                err_reuse_ALL_dhda = np.mean(dhda_reuse_ALL_error_list)
                dhda_reuse_ALL_y.append(err_reuse_ALL_dhda)
                dhda_reuse_ALL_error_list = list()
    time1 = time.time()
    performance_dict["dhda_reuse_ALL_model"] = dhda_reuse_ALL_y
    time_cost_dict["dhda_reuse_ALL_model"] = time1 - time0

    logging.info("--------dhda reuse model w/o upper--------")
    random.seed(seed)
    np.random.seed(seed)
    wo_model = DaL_Regressor(MODEL_REUSE=True, Upper_Adapt=False)
    # dhda_model.fit_model(train_x, np.ravel(train_y))
    wo_model.fit_model(train_x, train_y)

    dhda_reuse_ALL_error_list = []
    dhda_reuse_ALL_y = []

    c = 0
    time0 = time.time()
    for index, env_data in enumerate(test_data_list):
        env_test_x, env_test_y = get_attributes_result(env_data)
        test_size = len(env_test_y)

        for i in range(test_size):
            c += 1
            y_pre = wo_model.predict(env_test_x[i][np.newaxis, :])
            val = 0
            # val = abs(y_pre - y_test[i])[0][0] change
            if env_test_y[i] != 0:
                val = ((abs(y_pre - env_test_y[i])[0]) / env_test_y[i])[0]
                dhda_reuse_ALL_error_list.append(val)
                wo_model.learn_data(env_test_x[i][np.newaxis, :], env_test_y[i][np.newaxis, :], val)
            if c % 32 == 0:
                err_reuse_ALL_dhda = np.mean(dhda_reuse_ALL_error_list)
                dhda_reuse_ALL_y.append(err_reuse_ALL_dhda)
                dhda_reuse_ALL_error_list = list()
    time1 = time.time()
    performance_dict["dhda_no_upper"] = dhda_reuse_ALL_y
    time_cost_dict["dhda_no_upper"] = time1 - time0

    logging.info("--------dhda reuse model w/o lower--------")
    random.seed(seed)
    np.random.seed(seed)
    wo_model = DaL_Regressor(MODEL_REUSE=True, Lower_Adapt=False)
    # dhda_model.fit_model(train_x, np.ravel(train_y))
    wo_model.fit_model(train_x, train_y)

    dhda_reuse_ALL_error_list = []
    dhda_reuse_ALL_y = []

    c = 0
    time0 = time.time()
    for index, env_data in enumerate(test_data_list):

        env_test_x, env_test_y = get_attributes_result(env_data)
        test_size = len(env_test_y)

        for i in range(test_size):
            c += 1
            y_pre = wo_model.predict(env_test_x[i][np.newaxis, :])
            val = 0
            # val = abs(y_pre - y_test[i])[0][0] change
            if env_test_y[i] != 0:
                val = ((abs(y_pre - env_test_y[i])[0]) / env_test_y[i])[0]
                dhda_reuse_ALL_error_list.append(val)
                wo_model.learn_data(env_test_x[i][np.newaxis, :], env_test_y[i][np.newaxis, :], val)
            if c % 32 == 0:
                err_reuse_ALL_dhda = np.mean(dhda_reuse_ALL_error_list)
                dhda_reuse_ALL_y.append(err_reuse_ALL_dhda)
                dhda_reuse_ALL_error_list = list()
    time1 = time.time()
    performance_dict["dhda_no_lower"] = dhda_reuse_ALL_y
    time_cost_dict["dhda_no_lower"] = time1 - time0

    logging.info("--------dhda reuse model w/o hybrid update--------")
    random.seed(seed)
    np.random.seed(seed)
    wo_model = DaL_Regressor(MODEL_REUSE=True, HYBRID_UPDATE=False)
    # dhda_model.fit_model(train_x, np.ravel(train_y))
    wo_model.fit_model(train_x, train_y)

    dhda_reuse_ALL_error_list = []
    dhda_reuse_ALL_y = []

    c = 0
    time0 = time.time()
    for index, env_data in enumerate(test_data_list):
        env_test_x, env_test_y = get_attributes_result(env_data)
        test_size = len(env_test_y)

        for i in range(test_size):
            c += 1
            y_pre = wo_model.predict(env_test_x[i][np.newaxis, :])
            val = 0
            # val = abs(y_pre - y_test[i])[0][0] change
            if env_test_y[i] != 0:
                val = ((abs(y_pre - env_test_y[i])[0]) / env_test_y[i])[0]
                dhda_reuse_ALL_error_list.append(val)
                wo_model.learn_data(env_test_x[i][np.newaxis, :], env_test_y[i][np.newaxis, :], val)
            if c % 32 == 0:
                err_reuse_ALL_dhda = np.mean(dhda_reuse_ALL_error_list)
                dhda_reuse_ALL_y.append(err_reuse_ALL_dhda)
                dhda_reuse_ALL_error_list = list()
    time1 = time.time()
    performance_dict["dhda_no_hybrid"] = dhda_reuse_ALL_y
    time_cost_dict["dhda_no_hybrid"] = time1 - time0

    logging.info("--------dhda reuse model w/o reuse--------")
    random.seed(seed)
    np.random.seed(seed)
    wo_model = DaL_Regressor(MODEL_REUSE=True, CPD=False, Smooth=False)
    # dhda_model.fit_model(train_x, np.ravel(train_y))
    wo_model.fit_model(train_x, train_y)

    dhda_reuse_ALL_error_list = []
    dhda_reuse_ALL_y = []

    c = 0
    time0 = time.time()
    for index, env_data in enumerate(test_data_list):
        env_test_x, env_test_y = get_attributes_result(env_data)
        test_size = len(env_test_y)

        for i in range(test_size):
            c += 1
            y_pre = wo_model.predict(env_test_x[i][np.newaxis, :])
            val = 0
            # val = abs(y_pre - y_test[i])[0][0] change
            if env_test_y[i] != 0:
                val = ((abs(y_pre - env_test_y[i])[0]) / env_test_y[i])[0]
                dhda_reuse_ALL_error_list.append(val)
                wo_model.learn_data(env_test_x[i][np.newaxis, :], env_test_y[i][np.newaxis, :], val)
            if c % 32 == 0:
                err_reuse_ALL_dhda = np.mean(dhda_reuse_ALL_error_list)
                dhda_reuse_ALL_y.append(err_reuse_ALL_dhda)
                dhda_reuse_ALL_error_list = list()
    time1 = time.time()
    performance_dict["dhda_no_cpdsmooth"] = dhda_reuse_ALL_y
    time_cost_dict["dhda_no_cpdsmooth"] = time1 - time0
    return performance_dict, time_cost_dict




def set_logging(filename):
    # work_dir = os.path.join(PROJECT_ROOT, "result", "reuse")
    # 构造日志文件路径
    save_path = os.path.join(work_dir, "log", f"{filename}_log.txt")
    # 获取根日志器
    logger = logging.getLogger()
    # 移除所有已存在的处理器（避免重复输出到之前的文件）
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 创建新的文件处理器
    file_handler = logging.FileHandler(save_path, mode='w')
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    # 设置日志级别
    logger.setLevel(logging.INFO)
    # 添加新的处理器
    logger.addHandler(file_handler)


ENV_INDEX2NAME = {0: "sac", 1: "x264", 2: "storm", 3: "spear", 4: "sqlite", 5: "nginx", 6: "exastencils", 7: "deeparch", 8: "lighttpd", 9: "mysql", 10: "imagemagick", 11: "libvips", 12: "pulsar", 13: "rabbitmq"}

if __name__ == '__main__':

    random.seed(seed)
    np.random.seed(seed)

    envs = [env_list1, env_list2, env_list3, env_list4,
            env_list5, env_list6, env_list7, env_list8]

    for i, env in enumerate(envs):

        # Preserve the old 1-3 / 4-6 / 7-8 script boundaries.
        if i in {0, 3, 6}:
            seed = 43
            random.seed(seed)
            np.random.seed(seed)
        log_filename = ENV_INDEX2NAME[i]
        csv_save_path = os.path.join(work_dir, "csv", f"{log_filename}_result.csv")

        # 如果已有旧文件，可以先删掉（可选）
        if os.path.exists(csv_save_path):
            os.remove(csv_save_path)

        for index in range(30):

            seed += 2 * index + 1
            random.seed(seed)
            np.random.seed(seed)

            set_logging(log_filename + f"_{index}_test")

            logging.info(f"the {i}-th env is testing")
            logging.info(f"the {index}-th testing of the {i}-th env")

            performance_dict, time_cost_dict = test_dhda_reuse(
                log_filename,
                env
            )

            fig_save_path = os.path.join(work_dir, "figures") + os.sep
            fig_save_path_filename = (
                fig_save_path +
                f"{log_filename}_{index}_test_figure.png"
            )

            plot_show(
                performance_dict,
                save_fig=True,
                title=f"the {index} test of {log_filename}",
                save_path=fig_save_path_filename
            )

            # 当前一次实验结果
            current_result = {}

            for key, value in performance_dict.items():
                current_result[key] = np.mean(value)

            # 额外记录实验编号和seed（推荐）
            current_result["repeat_id"] = index
            current_result["seed"] = seed

            print(f"all method time cost: {time_cost_dict}")

            row_df = pd.DataFrame([current_result])

            # 首次写入带header，后续append
            row_df.to_csv(
                csv_save_path,
                mode='a',
                header=not os.path.exists(csv_save_path),
                index=False
            )
