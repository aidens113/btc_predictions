import pyautogui
import time
import math
import random
import os
import sys
import requests
import wmi
import imaplib
import email
from email.header import decode_header
import webbrowser
import threading
from os.path import expanduser
import concurrent.futures
from datetime import datetime
from seleniumwire import webdriver
from selenium.webdriver.common.keys import Keys
from os.path import expanduser
import concurrent.futures
from datetime import datetime
import time,string,zipfile,os
#import selenium

from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys

def initdriver():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_experimental_option("excludeSwitches", ['enable-automation'])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    
    chrome_options.add_argument("--headless")   
    
    #scheme='http'
   # )
    #chrome_options.add_extension(proxyauth_plugin_path)
    
    driver = webdriver.Chrome(executable_path='chromedriver.exe',options=chrome_options)
    driver.set_page_load_timeout(25)
    driver.delete_all_cookies()
    driver.set_window_position(-10000,0)
    return driver

while True:
    try:
        driver = initdriver()
        driver.get("https://www.basefex.com/trade/BTCUSDT")
        price = str(driver.title).split(" ")[0]        
        file = open("1minprices.txt","a")
        file.write(price+"\n")
        file.close()
        try:
            driver.close()
            driver.quit()
        except:
            print("Error fully closing driver")

        time.sleep(60)
    except Exception as EE:
        print("Error: "+str(EE))
