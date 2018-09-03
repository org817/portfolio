import requests
import pandas as pd
import time
import random
from bs4 import BeautifulSoup as soup
import lxml
import re
import os
import urllib.request
from multiprocessing import Process


def download_log(indi_list):
    for i, down_indi in enumerate(indi_list):
        filename = down_indi.split('/')[-1]
        print('start :', filename)
        try:
            urllib.request.urlretrieve('http://' + down_indi, "D:/seclogdown/" + filename)
            print('downloaded :', down_indi)
        except:
            failed.append(down_indi)
            print('--------------- error : ', down_indi)


if __name__ == '__main__':

    base_url = "https://www.sec.gov/files/EDGAR_LogFileData_thru_Jun2017.html"
    data_files = requests.get(base_url).text
    soup_obj = soup(data_files, 'lxml')
    data_list = soup_obj.find('body').get_text().splitlines()
    data_list = data_list[2:]

    print('total number of files :', len(data_list))
    data_list = data_list[4961:5122]
    print('data target:', data_list[0], '~', data_list[-1])

    par_data = []
    failed = []
    partition = 4
    total_len = len(data_list)
    part = round(total_len / partition)
    for i in range(partition):
        da = []
        if i == partition-1:
            da = data_list[i * part:]
        else:
            da = data_list[i * part:(i + 1) * part]

        par_data.append(da)

    procs = []
    for i in range(partition):
        proc = Process(target=download_log, args=(par_data[i],))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()