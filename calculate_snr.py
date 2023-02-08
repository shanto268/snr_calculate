# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 16:54:24 2022

@author: shanto
"""

import Labber
import os
import sys
from utilities import Watt2dBm, dBm2Watt, VNA2dBm
from snr_helper_functions import *

if __name__ == "__main__":
    labber_data_file = str(input("Labber File Location: "))
    repeated = int(input("Number of Repeations: "))

    std_highSNR = 1.15 # cut off point for determining high SNR
    cutOff_around_SA_peak = 10e3 # Hz

    lf = Labber.LogFile(labber_data_file)

    SA_channel_name = 'HP Spectrum Analyzer - Signal'

    signal = lf.getData(name = SA_channel_name)
    linsig = dBm2Watt(signal)

    SAxdata, SAydata = lf.getTraceXY(y_channel=SA_channel_name) # gives last trace from SA

    get_average_SNR(signal, repeated, SAxdata,cutOff)

