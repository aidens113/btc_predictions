import numpy
import matplotlib.pyplot as plt
import pandas
from pandas import DataFrame
import math
import yfinance as yf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import sys
import os
import time
import random
import requests
from datetime import datetime
import hashlib
import hmac
from urllib.parse import urlparse
import json


def generate_signature(secret, http_method, url, expires, data):
    # parse relative path
    parsedURL = urlparse(url)
    path = parsedURL.path
    if parsedURL.query:
        path = path + '?' + parsedURL.query

    if isinstance(data, (bytes, bytearray)):
        data = data.decode('utf8')

    print("Computing HMAC: %s" % http_method + path + str(expires) + data)
    message = http_method + path + str(expires) + data

    signature = hmac.new(bytes(secret, 'utf8'), bytes(message, 'utf8'), digestmod=hashlib.sha256).hexdigest()
    return signature


file = open("api.txt","r")
keys = file.read()
file.close()

apikey = keys.split(":")[0].strip().replace("\n","").replace("\r","")
apisecret = keys.split(":")[1].strip().replace("\n","").replace("\r","")



def cancelorder(theid):
    try:
        global apikey
        global apisecret
        for _ in range(3):
            timestamp = datetime.now().timestamp()
            expires = int(round(timestamp) + 5)    
            authkey = generate_signature(apisecret,'DELETE',str("https://api.basefex.com/orders/")+str(theid),expires, '')
            hed={'api-expires':str(expires),'api-key':apikey,'api-signature':authkey}
            requests.delete(str("https://api.basefex.com/orders/")+str(theid), headers=hed)
            time.sleep(1)        
    except:
        print("Random error, trying again")


def long(symbol,amount, price, numbuy, numsell):
    try:
        global apikey
        global apisecret
        args = {'size':str(amount),
                'symbol':str(symbol),
                'type':'LIMIT',
                'side':'BUY',
                'price':str(price)}

        timestamp = datetime.now().timestamp()
        expires = int(round(timestamp) + 5)    
        authkey = generate_signature(apisecret,'POST','https://api.basefex.com/orders',expires, json.dumps(args))
        hed={'api-expires':str(expires),'api-key':apikey,'api-signature':authkey}
        response = requests.post("https://api.basefex.com/orders", json=args, headers=hed)
        response = response.text
        print(response)
        time.sleep(3)
        numnewbuy = getopentrades(symbol,"NEW","BUY")
        numnewsell = getopentrades(symbol,"NEW","SELL")    
        
        theid = str(str(response).split('"id":"')[1].split('",')[0].strip().replace("\r","").replace("\n",""))
        for _ in range(3):
            try:
                time.sleep(2)
                print("Checking for trade finished")
                if tradesopen(symbol,"BUY",numbuy) == True or tradesopen(symbol,"SELL",numsell) == True:
                    print("long pos: Amount: "+str(amount)+" Symbol: "+str(symbol)+" Price: "+str(price))
                    return True
            except:
                print("Error longing, trying again")

        time.sleep(3)
        for _ in range(10):
            try:
                print("Error placing order in time. Cancelling")
                #Last check before cancelling
                if tradesopen(symbol,"BUY",numbuy) == True or tradesopen(symbol,"SELL",numsell) == True:
                    return True
                cancelorder(theid)
                for _ in range(3):
                    time.sleep(2)
                    print("Checking for trade cancelled")
                    if tradesnew(symbol,"BUY",numnewbuy) == True and tradesnew(symbol,"SELL",numnewsell) == True:
                        if tradesopen(symbol,"BUY",numbuy) == True or tradesopen(symbol,"SELL",numsell) == True:
                            return True
                        print("Successfully cancelled trade")
                        return False
            except:
                print("Error cancelling, trying again")                
    except:
        print("Random error, trying again")
    

def short(symbol,amount,price, numbuy, numsell):
    try:
        global apikey
        global apisecret
        args = {'size':str(amount),
                'symbol':str(symbol),
                'type':'LIMIT',
                'side':'SELL',
                'price':str(price)}
        
        timestamp = datetime.now().timestamp()
        expires = int(round(timestamp) + 5)    
        authkey = generate_signature(apisecret,'POST','https://api.basefex.com/orders',expires, json.dumps(args))
        hed={'api-expires':str(expires),'api-key':apikey,'api-signature':authkey}
        response = requests.post("https://api.basefex.com/orders", json=args, headers=hed)
        response = response.text
        print(response)
        time.sleep(3)
        numnewbuy = getopentrades(symbol,"NEW","BUY")
        numnewsell = getopentrades(symbol,"NEW","SELL") 
        
        theid = str(str(response).split('"id":"')[1].split('",')[0].strip().replace("\r","").replace("\n",""))
        for _ in range(3):
            try:
                time.sleep(2)
                print("Checking for trade finished")
                if tradesopen(symbol,"BUY",numbuy) == True or tradesopen(symbol,"SELL",numsell) == True:
                    print("short pos: Amount: "+str(amount)+" Symbol: "+str(symbol)+" Price: "+str(price))
                    return True
            except:
                print("Error shorting, trying again")
        time.sleep(3)
        for _ in range(10):
            try:
                print("Error placing order in time. Cancelling")
                #Last check before cancelling
                if tradesopen(symbol,"BUY",numbuy) == True or tradesopen(symbol,"SELL",numsell) == True:
                    return True
                cancelorder(theid)
                for _ in range(3):
                    time.sleep(2)
                    print("Checking for trade cancelled")
                    if tradesnew(symbol,"BUY",numnewbuy) == True and tradesnew(symbol,"SELL",numnewsell) == True:
                        if tradesopen(symbol,"BUY",numbuy) == True or tradesopen(symbol,"SELL",numsell) == True:
                            return True
                        print("Successfully cancelled trade")
                        return False
            except:
                print("Error cancelling, trying again")
    except:
        print("Random error, trying again")        
