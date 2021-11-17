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
#GETS NEW DATA FOR BTC PRICE FROM YAHOO FINANCE

crypto = "BTC-USD"
#crypto = "ETH-USD"

btc = yf.Ticker(crypto)

history = btc.history(period='1mo',interval="90m")

history.to_csv('out.csv')


#tickerdata = pandas.read_csv('BTCUSDT.csv')
#tickerdata = DataFrame(tickerdata)

#print(tickerdata.values[1])
#TESTING
futuretime = 1
def predictdata(tickerdata):
    global futuretime
    scaler = MinMaxScaler(feature_range = (0,1))
    scaler1 = MinMaxScaler(feature_range = (0,1))

    #i = 0
    dataset = DataFrame()
    dataset1 = DataFrame()
    for i in range(1,len(tickerdata)):
        if i <= int(len(tickerdata) * 0.6):
            dataset = dataset.append(tickerdata.iloc[i])
        if i >= int(len(tickerdata) * 0.6):
            dataset1 = dataset1.append(tickerdata.iloc[i])

    #file = open("1minprices.txt","r")
    #newdata = file.readlines()
    #file.close()
    #for item in newdata:
       

    dataset = DataFrame(dataset)
    dataset = scaler.fit_transform(dataset)

    dataset1 = DataFrame(dataset1)
    dataset1 = scaler1.fit_transform(dataset1)

    #PRINTS REAL DATA FOR COMPARISON
    print(dataset1[0])
    #plt.plot(dataset)
    #plt.plot(dataset1)
    #plt.show()

    #INITIATES NETWORK

    mind = Sequential()

    trainx = []
    trainy = []


    testx = []
    testy = []


    #AMOUNT OF TIME ALGO SHOULD SEE IN THE PAST
    #(IF DATA IS 1 DAY DATA, THEN 1 TIME STEP = 1 DAY)

    timesteps = 30


    #ADDS ITEMS TO TRAINING DATASET

    for i in range(timesteps, len(dataset)):
        trainx.append(dataset[i-timesteps:i, :])
        trainy.append(dataset[i])
        
    trainx = numpy.array(trainx)
    trainy = numpy.array(trainy)



    #ADDS ITEMS TO TEST DATASET

    for i in range(timesteps, len(dataset1)):
        testx.append(dataset1[i-timesteps:i, :])
        testy.append(dataset1[i])

    testx = numpy.array(testx)
    testy = numpy.array(testy)

    print(trainx.shape)


    #BUILDS AND COMPILES MODEL



    mind.add(LSTM(50, return_sequences=True,input_shape=(trainx.shape[1], trainx.shape[2]) ))
    mind.add(Dropout(0.6))

    mind.add(LSTM(50, return_sequences=True ))
    mind.add(Dropout(0.6))

    mind.add(LSTM(50, return_sequences=True ))
    mind.add(Dropout(0.6))

    mind.add(LSTM(50))
    mind.add(Dropout(0.6))

    mind.add(Dense(1,activation='linear'))
    mind.compile(loss='mean_squared_error', optimizer='adam')

    os.system('cls')

    #SAVE WEIGHTS

    #cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)

    #TRAINS ALGO

    mind.fit(trainx, trainy, epochs=5, batch_size=60)#,callbacks=[cp_callback]


    os.system('cls')

    #FEED IN TESTX (60 timesteps or days)


    #FOR LOOP THAT FEEDS PREDICTED NEW DATA BACK INTO DATASET
    #TO GET THE PREDICTED FORCAST

    datasettemp = dataset1
    for i in range(futuretime):
        trainprediction = mind.predict(testx)
        testx = []
        datasettemp = numpy.append(datasettemp,trainprediction[int(len(trainprediction) - 1)][0])
        datasettemp = datasettemp.reshape(datasettemp.shape[0], 1)
        print("Predicted Price: "+str(datasettemp[ int(len(datasettemp)-1) ]))
        for i in range(timesteps, len(datasettemp)):
            testx.append(datasettemp[i-timesteps:i, :])
        
        testx = numpy.array(testx)        



    #CONVERTS STANDARDIZED DATA TO NORMAL DATA

    trainprediction = scaler1.inverse_transform(trainprediction)

    datasettocompare = scaler1.inverse_transform(dataset1)
    return trainprediction, datasettocompare

#COMPARES TODAY'S ESTIMATED PRICE AND X DAY'S PREDICTED PRICE TO GET
#PREDICTED PRICE MOVEMENT


#BUY AND SELL API
#30 BTCUSD = 1 BTCUSDT

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

def getopentrades(symbol, status, side):
    try:
        global apikey
        global apisecret
        timestamp = datetime.now().timestamp()
        expires = int(round(timestamp) + 5)    
        authkey = generate_signature(apisecret,'GET',str('/orders/count?status='+str(status)+'&side='+str(side)+'&symbol='+str(symbol)),expires, '')
        hed={'api-expires':str(expires),'api-key':apikey,'api-signature':authkey}
        response = requests.get("https://api.basefex.com/orders/count?status="+str(status)+"&side="+str(side)+"&symbol="+str(symbol), headers=hed)
        print(response.text)
        orders = str(str(response.text).split('"count":')[1].split("}")[0].strip())
        orders = int(orders)
        return orders
    except:
        print("Random error, trying again")

def tradesopen(symbol, side,previousamount):
    try:
        newamount = getopentrades(symbol,"FILLED",side)
        tradeson = None
        if newamount > previousamount:
            tradeson = True
        else:
            tradeson = False
        return tradeson
    except:
        print("Random error, trying again")

def tradesnew(symbol, side,previousamount):
    try:
        newamount = getopentrades(symbol,"NEW",side)
        tradeson = None
        if newamount < previousamount or int(newamount) == 0:
            tradeson = True
        else:
            tradeson = False
        return tradeson
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


def closelong(symbol,amount, price, numbuy, numsell):
    try:
        global apikey
        global apisecret
        args = {'size':str(amount),
                'symbol':str(symbol),
                'type':'MARKET',
                'side':'SELL',
                'price':str(price)}

        timestamp = datetime.now().timestamp()
        expires = int(round(timestamp) + 5)    
        authkey = generate_signature(apisecret,'POST','https://api.basefex.com/orders',expires, json.dumps(args))
        hed={'api-expires':str(expires),'api-key':apikey,'api-signature':authkey}
        requests.post("https://api.basefex.com/orders", json=args, headers=hed)
        return True
    except:
        print("Random error, trying again")

def closeshort(symbol,amount,price, numbuy, numsell):
    try:
        global apikey
        global apisecret
        args = {'size':str(amount),
                'symbol':str(symbol),
                'type':'MARKET',
                'side':'BUY',
                'price':str(price)}
        
        timestamp = datetime.now().timestamp()
        expires = int(round(timestamp) + 5)    
        authkey = generate_signature(apisecret,'POST','https://api.basefex.com/orders',expires, json.dumps(args))
        hed={'api-expires':str(expires),'api-key':apikey,'api-signature':authkey}
        requests.post("https://api.basefex.com/orders", json=args, headers=hed)
        return True
    except:
        print("Random error, trying again")

    
def getmarketprice(contract):
    global apikey
    global apisecret
    for _ in range(5):
        try:
            response = requests.get("https://api.basefex.com/instruments/prices")
            price = str(str(response.text).split(contract)[1].split(',"price":')[1].split(".")[0])
            return int(price)
        except Exception as WW:
            print("Exception with market price: "+str(WW))


#PREDICTS DATA
def commitpredict(col):
    tickerdata = pandas.read_csv('out.csv',usecols=[col,])
    tickerdata = DataFrame(tickerdata)
    predict,realdata = predictdata(tickerdata)
    return predict,realdata

def backpredict(col,daynum):
    tickerdata = pandas.read_csv('out.csv',usecols=[col,])
    tickerdata = DataFrame(tickerdata)

    dataset = DataFrame()
    for i in range(1,len(tickerdata)):
        if i <= int(len(tickerdata) - daynum - 1):
            dataset = dataset.append(tickerdata.iloc[i])
        
    predict,realdata = predictdata(dataset)
    return predict,realdata

def makedataset(col,daynum):
    tickerdata = pandas.read_csv('out.csv',usecols=[col,])
    tickerdata = DataFrame(tickerdata)

    dataset = []
    for i in range(1,len(tickerdata)):
        if i <= int(len(tickerdata) - daynum - 1):
            dataset.append(float(tickerdata.iloc[i]))
    return dataset

def printdata1(trainprediction, label):
    global futuretime
    print("Today's predicted "+str(label)+": "+str(trainprediction[int(len(trainprediction) - futuretime - 1)]))
    print(str(futuretime)+" days predicted "+str(label)+": "+str(trainprediction[int(len(trainprediction) - 1)]))
    upordown = '' #True = Up, False = Down
    difference = 0

    if float(str(trainprediction[int(len(trainprediction) - 1)]).replace("[","").replace("]","")) > float(str(trainprediction[int(len(trainprediction) - futuretime - 1)]).replace("]","").replace("[","") ):
        upordown = 'Up'
        difference = float(str(trainprediction[int(len(trainprediction) - futuretime - 1)]).replace("]","").replace("[","") ) - float(str(trainprediction[int(len(trainprediction) - 1)]).replace("[","").replace("]",""))
    else:
        upordown = 'Down'
        difference = float(str(trainprediction[int(len(trainprediction) - 1)]).replace("[","").replace("]","")) - float(str(trainprediction[int(len(trainprediction) - futuretime - 1)]).replace("]","").replace("[","") )

    difference = float(str(difference).replace("-",""))

    #PRINTS FINAL ESTIMATED AND REAL DATA
    if "close" in label:
        print(" ")
        print("-------------------------------------------------------------------")
        print("Difference between "+str(label)+": "+str(difference) )
        print("Predicted Movement from predicted today to predicted day "+str(futuretime)+": "+str(upordown))
        print(" ")
        print("-------------------------------------------------------------------")

def printdata2(trainprediction, datasettocompare, label):
    global futuretime
    upordown = '' #True = Up, False = Down
    difference = 0

    if float(str(trainprediction[int(len(trainprediction) - 1)]).replace("[","").replace("]","")) > float(str(trainprediction[int(len(trainprediction) - futuretime - 1)]).replace("]","").replace("[","") ):
        upordown = 'Up'
        difference = float(str(trainprediction[int(len(trainprediction) - futuretime - 1)]).replace("]","").replace("[","") ) - float(str(trainprediction[int(len(trainprediction) - 1)]).replace("[","").replace("]",""))
    else:
        upordown = 'Down'
        difference = float(str(trainprediction[int(len(trainprediction) - 1)]).replace("[","").replace("]","")) - float(str(trainprediction[int(len(trainprediction) - futuretime - 1)]).replace("]","").replace("[","") )

    difference = float(str(difference).replace("-",""))

    #PRINTS FINAL ESTIMATED AND REAL DATA
    
    print("Today's Actual "+str(label)+": "+str(str(datasettocompare[int(len(datasettocompare) - 1)])))
    if upordown == "Up":
        print(str(futuretime)+" Days Predicted "+str(label)+": "+str(float( datasettocompare[int(len(datasettocompare) - 1)] + difference)  ))
    else:
        print(str(futuretime)+" Days Predicted "+str(label)+": "+str(float( datasettocompare[int(len(datasettocompare) - 1)] - difference)  ))
    print("--")


def printdata3(trainprediction, datasettocompare,high,low, label):
    global futuretime
    upordown = '' #True = Up, False = Down
    difference = 0

    #IF LOW OF PREDICTED DAY IS LESS THAN PREDICTION PRICE, SUBTRACT PREDICTION FROM HIGH OF DAY (TO GET MAX POSSIBLE DIFFERENCE). 
    if low < float(str(trainprediction[int(len(trainprediction) - futuretime - 1)]).replace("]","").replace("[","") ):
        upordown = 'Up'
        difference = float(str(high).replace("]","").replace("[","")) - float(str(trainprediction[int(len(trainprediction) - futuretime - 1)]).replace("]","").replace("[",""))
    #OTHERWISE SUBTRACT LOW OF PREDICTED DAY FROM PRICE PREDICTION TO GET MINIMUM PRICE IN PREDICTED DAY (TO GET MAX POSSIBLE DIFFERENCE)
    else:
        upordown = 'Down'
        difference = float(str(trainprediction[int(len(trainprediction) - futuretime - 1)]).replace("]","").replace("[","")) - float(str(low).replace("]","").replace("[",""))

    difference = float(str(difference).replace("-",""))

    #PRINTS FINAL ESTIMATED AND REAL DATA
    print(" ")
    print("-------------------------------------------------------------------")
    print("Today's Actual "+str(label)+": "+str(str(datasettocompare[int(len(datasettocompare) - 1)])))
    if upordown == "Up":
        print(str(futuretime)+" Days Predicted "+str(label)+": "+str(float( datasettocompare[int(len(datasettocompare) - 1)] + difference)  ))
    else:
        print(str(futuretime)+" Days Predicted "+str(label)+": "+str(float( datasettocompare[int(len(datasettocompare) - 1)] - difference)  ))

    if "close" in label:
        print(" ")
        print("-------------------------------------------------------------------")
        print("Predicted Total Price Movement From Now To Then "+str(label)+": "+str(difference) )
        print("Predicted Movement from predicted today to predicted day "+str(futuretime)+": "+str(upordown))
        print("Time To Buy: Always At End Of Current Day (Day Close Price). Make Predictions Every 24hrs")
        print(" ")
        print("-------------------------------------------------------------------")
        


def decidetradeoption(trainprediction, datasettocompare,high,low, label):
    global futuretime
    upordown = '' #True = Up, False = Down
    difference = 0

    #IF LOW OF PREDICTED DAY IS LESS THAN PREDICTION PRICE, SUBTRACT PREDICTION FROM HIGH OF DAY (TO GET MAX POSSIBLE DIFFERENCE). 
    if low < float(str(trainprediction[int(len(trainprediction) - futuretime - 1)]).replace("]","").replace("[","") ):
        upordown = 'Up'
        difference = float(str(high).replace("]","").replace("[","")) - float(str(trainprediction[int(len(trainprediction) - futuretime - 1)]).replace("]","").replace("[",""))
    #OTHERWISE SUBTRACT LOW OF PREDICTED DAY FROM PRICE PREDICTION TO GET MINIMUM PRICE IN PREDICTED DAY (TO GET MAX POSSIBLE DIFFERENCE)
    else:
        upordown = 'Down'
        difference = float(str(trainprediction[int(len(trainprediction) - futuretime - 1)]).replace("]","").replace("[","")) - float(str(low).replace("]","").replace("[",""))

    difference = float(str(difference).replace("-",""))

    #PRINTS FINAL ESTIMATED AND REAL DATA
    print(" ")
    print("-------------------------------------------------------------------")
    print("Today's Actual "+str(label)+": "+str(str(datasettocompare[int(len(datasettocompare) - 1)])))
    if upordown == "Up":
        print(str(futuretime)+" Days Predicted "+str(label)+": "+str(float( datasettocompare[int(len(datasettocompare) - 1)] + difference)  ))
    else:
        print(str(futuretime)+" Days Predicted "+str(label)+": "+str(float( datasettocompare[int(len(datasettocompare) - 1)] - difference)  ))

    
    print("-------------------------------------------------------------------")
    print("Predicted Total Price Movement From Now To Then "+str(label)+": "+str(difference) )
    print("Predicted Movement from predicted today to predicted day "+str(futuretime)+": "+str(upordown))
    print("Time To Buy: Always At End Of Current Day (Day Close Price). Make Predictions Every 24hrs")
    print(" ")
    print("-------------------------------------------------------------------")
    return upordown, difference



    

def doubletrade():
    if "y" in str(input("Double Trade(recommended)? y/n/alt: ")).lower():
        doubletrade = True
        print("Double trading activated")
    else:
        doubletrade = False
        print("Double trading off")
    usdtcontracts = 10
    if doubletrade == True: 
        btcusdcontracts = int(usdtcontracts * 20)    
    else:
        btcusdcontracts = int(usdtcontracts * 20)
        usdtcontracts = int(usdtcontracts * 20)
    points = 0
    #MAIN LOOP
    while True:
        #GETS NEW INFO
        btc = yf.Ticker(crypto)

        history = btc.history(period='1mo',interval="90m")

        history.to_csv('out.csv')

        totalprofit = 0
        totalloss = 0

        
        #GETS PRICES

        if doubletrade == True:
            longcrypto = "BTCUSD"
            shortcrypto = "BTCUSDT"
        else:
            longcrypto = "BTCUSD"
            shortcrypto = "BTCUSD"
        
        price = float(getmarketprice(longcrypto)) + 15.0
        longstart = price
        templongstart = longstart

        
        price = float(getmarketprice(shortcrypto)) - 15.0
        shortstart = price
        tempshortstart = shortstart
        

        longprofit = -999999999999999
        longloss = float(float(longstart - 3500) )
        numbuylongsopen = getopentrades(longcrypto,"FILLED","BUY")
        numselllongsopen = getopentrades(shortcrypto,"FILLED","SELL")
        #LONG POSITION
        if doubletrade == False:
            print("Doing prediction")
            predictopendata,opendata = backpredict(1,0)
            predicthighdata,highdata = backpredict(2,0)
            predictlowdata,lowdata = backpredict(3,0)
            predictclosedata,closedata = backpredict(4,0)
            os.system('cls')
            #MODEL ASSUMES AUTOMATIC TRAILING STOP LOSS
        
            positiontohold,profitmargin = decidetradeoption(predictclosedata,closedata,predicthighdata[int(len(predicthighdata) - 1)],predictlowdata[int(len(predictlowdata) - 1)],'close')
        else:
            positiontohold = ""
        
        longopen = False
        if positiontohold == "Up" or doubletrade == True:
            while True:
                if long(longcrypto,btcusdcontracts,longstart,numbuylongsopen,numselllongsopen) == True:
                    longopen = True
                    break
                else:
                    price = float(getmarketprice(longcrypto)) + 15.0
                    longstart = price
                    templongstart = longstart
                    
        shortprofit = 999999999999999
        shortloss = float(float(shortstart + 3500) )
        
        numshortsopen = 0
        numbuyshortsopen = getopentrades(shortcrypto,"FILLED","BUY")
        numsellshortsopen = getopentrades(shortcrypto,"FILLED","SELL")
        shortopen = False
        #SHORT POSITION
        if positiontohold == "Down" or doubletrade == True:
            while True:
                if short(shortcrypto,usdtcontracts,shortstart,numbuyshortsopen,numsellshortsopen) == True:
                    shortopen = True
                    break
                else:
                    price = float(getmarketprice(shortcrypto)) - 15.0
                    shortstart = price
                    tempshortstart = shortstart

        #SECONDARY LOOP
        #Temporarily not in action - testing og method: NO NEED FOR TIMED LOOP AS MULTIPLE TRADES OPEN UP IN A WINDOW. STOP LOSSES AND TAKE PROFITS WILL AUTO ADJUST
        for _ in range(3300):
            try:
                time.sleep(0.5)
                longprice = float(getmarketprice(longcrypto))
                shortprice = float(getmarketprice(shortcrypto))
                os.system('cls')
                if doubletrade == True:
                    print("Long Startprice: "+str(longstart)+" Long Price: "+str(longprice)+" Long Take Profit: "+str(longprofit)+" Long Stop Loss: "+str(longloss)+" Long Active: "+str(longopen))
                    print("Short Startprice: "+str(shortstart)+" Short Price: "+str(shortprice)+" Short Take Profit: "+str(shortprofit)+" Short Stop Loss: "+str(shortloss)+" Short Active: "+str(shortopen))
                else:
                    if positiontohold == "Up":
                        print("Long Startprice: "+str(longstart)+" Long Price: "+str(longprice)+" Long Take Profit: "+str(longprofit)+" Long Stop Loss: "+str(longloss)+" Long Active: "+str(longopen))
                    if positiontohold == "Down":
                        print("Short Startprice: "+str(shortstart)+" Short Price: "+str(shortprice)+" Short Take Profit: "+str(shortprofit)+" Short Stop Loss: "+str(shortloss)+" Short Active: "+str(shortopen))
                #opens further trades if the difference between initial starting price
                #and current price is right
         
                #Calculates long position stats
                resetprofit = False
                if longopen == True:
                    #If trade is losing but only half as much loss as profit, exit trade
                    if doubletrade == True:
                        if float(longprice) <= float(longstart) and shortopen == False:
                            if float(longstart - longprice) <= float(totalprofit / 2.0):
                                print("Closing out remaining trade due to current position. Made profit")
                                while True:
                                    if closelong(longcrypto,btcusdcontracts,float(longprice - 15.0),numbuylongsopen,numselllongsopen) == True:
                                        break
                                    
                                totalloss += float(longstart - longprice)
                                points -= float(str(totalloss).replace("-","").strip())
                                file = open("realtradinginfo.txt","w")
                                file.write(str("Total Points: "+ str(float(points))+" Loss this trade: "+str(totalloss)+" Profit this trade: "+str(totalprofit)))
                                file.close()
                                
                                file = open("realtradinginfolog.txt","a")
                                file.write(str("Profit greater than 2x the loss: Total Points: "+ str(float(points))+" Loss this trade: "+str(totalloss)+" Profit this trade: "+str(totalprofit)+"\n"))
                                file.close()
                                resetprofit = True
                                longopen = False
                                
                    
                            
                    #Prices surpasses take profit (makes sure price isn't too low)
                    if float(longprice) <= float(longprofit) and float(longprice) >= float(longstart + 30): 
                        while True:
                            if closelong(longcrypto,btcusdcontracts,float(longprice - 15.0),numbuylongsopen,numselllongsopen) == True:
                                break
                        print("Made profit: "+str(float(longprice - longstart) ))
                        
                        if "-" not in str((longprice - longstart)):
                            points += float(longprice - longstart)
                            totalprofit += float(longprice - longstart)
                            file = open("realtradinginfo.txt","w")
                            file.write(str("Total Points: "+ str(float(points))+" Loss this trade: "+str(totalloss)+" Profit this trade: "+str(totalprofit)))
                            file.close()
                                
                            file = open("realtradinginfolog.txt","a")
                            file.write(str("Take profit: Total Points: "+ str(float(points))+" Loss this trade: "+str(totalloss)+" Profit this trade: "+str(totalprofit)+"\n"))
                            file.close()
                            
                            longopen = False     

                        
                    
                    #Trails price by 25 every time
                    if float(longprice) >= float(templongstart + 50):
                        longprofit = float(longprice - 15)
                        #Small interval of trailing after price
                        templongstart += 10
                        
                        
                    #stoploss
                    if float(longprice) <= float(longloss):
                        while True:
                            if closelong(longcrypto,btcusdcontracts,float(longprice - 15.0),numbuylongsopen,numselllongsopen) == True:
                                break
                        
                        totalprofit += float(longstart - longprice)
                        totalloss += float(longstart - longprice)
                        points -= float(str(totalloss).replace("-","").strip())
                        print("Hit stop loss on BTCUSD, closing contract")
                        file = open("realtradinginfo.txt","w")
                        file.write(str("Total Points: "+ str(float(points))+" Loss this trade: "+str(totalloss)+" Profit this trade: "+str(totalprofit)))
                        file.close()
                            
                        file = open("realtradinginfolog.txt","a")
                        file.write(str("Stop loss: Total Points: "+ str(float(points))+" Loss this trade: "+str(totalloss)+" Profit this trade: "+str(totalprofit)+"\n"))
                        file.close()
                        totalprofit = 0
                        totalloss = 0
                        longopen = False

                if resetprofit == True:
                    totalprofit = 0
                    totalloss = 0

                #Calculates short position stats
                
                if shortopen == True:
                    #If trade is losing but only half as much loss as profit, exit trade
                    if doubletrade == True:
                        if float(shortprice) >= float(shortstart) and longopen == False:
                            if float(shortprice - shortstart) <= float(totalprofit / 2.0):
                                print("Closing out remaining trade due to current position. Made profit")
                                while True:
                                    if closeshort(shortcrypto,usdtcontracts,float(shortprice + 15.0),numbuyshortsopen,numsellshortsopen) == True:
                                        break
                                
                                totalloss -= float(shortprice - shortstart)
                                points -= float(str(totalloss).replace("-","").strip())
                                file = open("realtradinginfo.txt","w")
                                file.write(str("Total Points: "+ str(float(points))+" Loss this trade: "+str(totalloss)+" Profit this trade: "+str(totalprofit)))
                                file.close()
                                
                                file = open("realtradinginfolog.txt","a")
                                file.write(str("Profit greater than 2x the loss: Total Points: "+ str(float(points))+" Loss this trade: "+str(totalloss)+" Profit this trade: "+str(totalprofit)+"\n"))
                                file.close()
                                totalprofit = 0
                                totalloss = 0
                                shortopen = False
                
                    #Prices surpasses take profit (makes sure price isn't too high)
                    if float(shortprice) >= float(shortprofit) and float(shortprice) <= float(shortstart - 30): 
                        while True:
                            if closeshort(shortcrypto,usdtcontracts,float(shortprice + 15.0),numbuyshortsopen,numsellshortsopen) == True:
                                break
                        print("Made profit: "+str(float(shortstart - shortprice) ))
                        
                        if "-" not in str((shortstart - shortprice)):
                            totalprofit += float(shortstart - shortprice)
                            points += float(shortstart - shortprice)
                            file = open("realtradinginfo.txt","w")
                            file.write(str("Total Points: "+ str(float(points))+" Loss this trade: "+str(totalloss)+" Profit this trade: "+str(totalprofit)))
                            file.close()
                            
                            file = open("realtradinginfolog.txt","a")
                            file.write(str("Take profit: Total Points: "+ str(float(points))+" Loss this trade: "+str(totalloss)+" Profit this trade: "+str(totalprofit)+"\n"))
                            file.close()

                            shortopen = False
                            
                    #Trails price by 25 every time
                    if float(shortprice) <= float(tempshortstart - 50):
                        shortprofit = float(shortprice + 15)
                        #Small interval of trailing after price
                        tempshortstart -= 10

                    #stoploss
                    if float(shortprice) >= float(shortloss):
                        while True:
                            if closeshort(shortcrypto,usdtcontracts,float(shortprice + 15.0),numbuyshortsopen,numsellshortsopen) == True:
                                break
                        totalprofit -= float(shortprice - shortstart)
                        totalloss -= float(shortprice - shortstart)
                        points -= float(str(totalloss).replace("-","").strip())
                        print("Hit stop loss on BTCUSD, closing contract")
                        file = open("realtradinginfo.txt","w")
                        file.write(str("Total Points: "+ str(float(points))+" Loss this trade: "+str(totalloss)+" Profit this trade: "+str(totalprofit)))
                        file.close()
                            
                        file = open("realtradinginfolog.txt","a")
                        file.write(str("Stop loss: Total Points: "+ str(float(points))+" Loss this trade: "+str(totalloss)+" Profit this trade: "+str(totalprofit)+"\n"))
                        file.close()
                        totalprofit = 0
                        totalloss = 0
                        shortopen = False

                if resetprofit == True:
                    totalprofit = 0
                    totalloss = 0      

                if longopen == False and shortopen == False:
                    file = open("realtradinginfolog.txt","a")
                    file.write("Ending both trades log \n")
                    file.close()
                    break

                #OPENS NEW TRADES - Temporarily not available due to testing og method
                #if longopen == False:
                    #Distance from short position to long
                    #if float(str(float(shortstart - longprice)).replace("-","").strip()) <= 300:
                        #file = open("realtradinginfolog.txt","a")
                        #file.write(str("Opening extra trade: "+ str(float(str(float(shortstart - longprice)).replace("-","").strip()))+"\n"))
                        #file.close()
                        #price = float(getmarketprice("BTCUSD")) + 15.0
                        #longstart = price
                        #templongstart = longstart
                        #numbuylongsopen = getopentrades("BTCUSD","FILLED","BUY")
                        #numselllongsopen = getopentrades("BTCUSD","FILLED","SELL")
                        #while True:
                            #if long('BTCUSD',btcusdcontracts,price,numbuylongsopen,numselllongsopen) == True:
                                #break
                            #else:
                                #price = float(getmarketprice("BTCUSD")) + 15.0
                                #longstart = price
                                #templongstart = longstart
                                
                        #longprofit = -999999999999999
                        #Longloss is bigger on larger trade gaps
                        #longloss = float(float(longstart - 1200) - float(str(float(shortstart - longprice)).replace("-","").strip()) - 50)
                        #resetprofit = True
                        #longopen = True
                    
                
                #if shortopen == False:
                    #Distance from short position to long
                    #if float(str(float(longstart - shortprice)).replace("-","").strip()) <= 300:
                        #file = open("realtradinginfolog.txt","a")
                        #file.write(str("Opening extra trade: "+ str(float(str(float(longstart - shortprice)).replace("-","").strip()))+"\n"))
                        #file.close()
                        #price = float(getmarketprice("BTCUSDT")) - 15.0
                        #shortstart = price
                        #tempshortstart = shortstart
                        #numbuyshortsopen = getopentrades("BTCUSDT","FILLED","BUY")
                        #numsellshortsopen = getopentrades("BTCUSDT","FILLED","SELL")
                        #while True:
                            #if short('BTCUSDT',usdtcontracts,price,numbuyshortsopen,numsellshortsopen) == True:
                                #break
                           #else:
                                #price = float(getmarketprice("BTCUSDT")) - 15.0
                                #shortstart = price
                                #tempshortstart = shortstart
                                
                        #shortprofit = 999999999999999
                        #Shortloss is bigger on larger trade gaps
                        #shortloss = float(float(shortstart + 1200) + float(str(float(longstart - shortprice)).replace("-","").strip()) + 50)
                        #resetprofit = True
                        #shortopen = True
            except Exception as EE:
                print("Random error, trying again: "+str(EE))            

        breakl = False
        
        #Closes all trades at the end of the 24 hour time limit
        if doubletrade == False:
            if longopen == True:
                while True:
                    if closelong(longcrypto,btcusdcontracts,float(longprice - 15.0),numbuylongsopen,numselllongsopen) == True:
                        file = open("realtradinginfo.txt","w")
                        file.write(str("Total Points: "+ str(float(points))+" Loss this trade: "+str(totalloss)+" Profit this trade: "+str(totalprofit)))
                        file.close()
                        
                        file = open("realtradinginfolog.txt","a")
                        file.write(str("Total Points: "+ str(float(points))+" Loss this trade: "+str(totalloss)+" Profit this trade: "+str(totalprofit)+"\n"))
                        file.close()
                        totalprofit = 0
                        totalloss = 0
                        breakl = True
                        break
                if breakl == True:
                    break
            if shortopen == True:
                while True:
                    if closeshort(shortcrypto,usdtcontracts,float(shortprice + 15.0),numbuyshortsopen,numsellshortsopen) == True:
                        file = open("realtradinginfo.txt","w")
                        file.write(str("Total Points: "+ str(float(points))+" Loss this trade: "+str(totalloss)+" Profit this trade: "+str(totalprofit)))
                        file.close()
                        
                        file = open("realtradinginfolog.txt","a")
                        file.write(str("Total Points: "+ str(float(points))+" Loss this trade: "+str(totalloss)+" Profit this trade: "+str(totalprofit)+"\n"))
                        file.close()
                        totalprofit = 0
                        totalloss = 0
                        breakl = True
                        break
                if breakl == True:
                    break

        


def backtrade(days, risk, leverage):
    #NEW IMPLIMENTATION: HEDGE 2 TRADES AGAINST EACHOTHER. TREAT THEM BOTH NORMALLY. LOSER MAKES PROFIT USUALLY OR AT LEAST BREAKS EVEN/LOSES LITTLE. WINNER STILL TAKES BIG PROFIT EVERY TIME.
    #BET 1.5X ON PREDICTED MOVEMENT
    
    maxspread = 0
    fee = 0.10
    lasttrade = ""
    tradeamount = 0.0
    balance = float(input("Balance to start with: "))
    if "y" in str(input("Double Trade(recommended)? y/n/alt: ")).lower():
        doubletrade = True
        print("Double trading activated")
    else:
        doubletrade = False
        
    totalprofit = 0
    totalloss = 0
    successful = 0
    failed = 0
    for day in range(-days,0):
        day = int(str(day).replace("-",""))
        print("Day: "+str(day))
        if doubletrade == False:
            predictopendata,opendata = backpredict(1,day)
            predicthighdata,highdata = backpredict(2,day)
            predictlowdata,lowdata = backpredict(3,day)
            predictclosedata,closedata = backpredict(4,day)
            #predictvolumedata,volumedata = backpredict(5,day)
            os.system('cls')

            #MODEL ASSUMES AUTOMATIC TRAILING STOP LOSS
            
            positiontohold,profitmargin = decidetradeoption(predictclosedata,closedata,predicthighdata[int(len(predicthighdata) - 1)],predictlowdata[int(len(predictlowdata) - 1)],'close')
                
        else:
            
            positiontohold = "Up"
            opendata = makedataset(1,day)
            highdata = makedataset(2,day)
            lowdata = makedataset(3,day)
            closedata = makedataset(4,day)
            
            
        #plt.clf()
        #plt.plot(closedata,label="Closes")
        profit = 0
        loss = 0
        #DETERMINES PROFIT/LOSS FROM PREVIOUS DAY'S TRADE
        #IF LAST TRADE IS LONG, AND TODAY'S HIGH IS HIGHER THAN YESTERDAY'S CLOSE, PROFIT WAS MADE
        slippage = float(float(tradeamount) * 0.005)
        fee = float(float(tradeamount)*0.0006)
        print("fee: "+str(fee))
        print("slippage: "+str(slippage))
        print("tradeamount: "+str(tradeamount))
        if lasttrade != "":    
            if lasttrade == "long" or doubletrade == True:
                #DEPENDING ON WHAT ALGORITHIM ADVISED, EITHER INCREASE OR DECREASE TRADE AMOUNT (SIMULATED AS MATH FOR PREVIOUS TRADE IS CALCULATED HERE)
                if doubletrade == True:
                    thetradeamount = tradeamount * 0.5
                else:
                    thetradeamount = tradeamount
                #if lasttrade == "long":
                    #thetradeamount = float(tradeamount * 0.75)
                #else:
                    #thetradeamount = float(tradeamount * 0.25)
                #PROFIT
                if highdata[int(len(highdata) - day - 1)] > closedata[int(len(closedata) - day - 2)]:
                    #PROFIT OF A FULL 1 COIN
                    profit = float(highdata[int(len(highdata) - day - 1)] - closedata[int(len(closedata) - day - 2)])
                    percentage = float(profit / closedata[int(len(closedata) - day - 1)]) + 1.0
                    print(percentage)
                    #MULTIPLY PRICE CHANGE PERCENTAGE BY TODAY'S CLOSE
                    profit = float(float(thetradeamount) * percentage) - thetradeamount
                    profit = (profit / 2)
                    profit = profit  * leverage
                    profit = profit - fee - slippage
                    balance += profit
                    totalprofit += profit
                    print("Gain long profit: "+str(profit))
                    
                #LOSS
                else:
                    
                    #LOSS OF COIN (TRADE MUST END BY END OF DAY)
                    loss = closedata[int(len(closedata) - day - 2)] - closedata[int(len(closedata) - day - 1)]
                    percentage = 1.0 - float(loss / closedata[int(len(closedata) - day - 1)])
                    print(percentage)
                    #MULTIPLY PRICE CHANGE PERCENTAGE BY TODAY'S CLOSE
                    loss = float(float(thetradeamount) * percentage) - thetradeamount
                    loss = loss  * leverage
                    loss = loss  + slippage  + fee
                    #LOSS IS A NEGATIVE NUMBER SO ADD IT TO BALANCE
                    balance += loss
                    totalloss += loss
                    print("Loss long profit: "+str(loss))
            
            if lasttrade == "short" or doubletrade == True:
                #DEPENDING ON WHAT ALGORITHIM ADVISED, EITHER INCREASE OR DECREASE TRADE AMOUNT (SIMULATED AS MATH FOR PREVIOUS TRADE IS CALCULATED HERE)
                if doubletrade == True:
                    thetradeamount = tradeamount * 0.5
                else:
                    thetradeamount = tradeamount
                #if lasttrade == "short":
                #    thetradeamount = float(tradeamount * 0.75)
                #else:
                #    thetradeamount = float(tradeamount * 0.25)
                #LOSS
                if lowdata[int(len(lowdata) - day - 1)] >= closedata[int(len(closedata) - day - 2)]:
                    
                    #LOSS OF COIN (TRADE MUST END BY END OF DAY)
                    loss = closedata[int(len(closedata) - day - 1)] - closedata[int(len(closedata) - day - 2)]
                    percentage = 1.0 -  float(loss / closedata[int(len(closedata) - day - 1)])
                    print(percentage)
                    #MULTIPLY PRICE CHANGE PERCENTAGE BY TODAY'S CLOSE
                    loss = float(float(thetradeamount) * percentage) - thetradeamount
                    loss = loss  * leverage
                    loss = loss  + slippage  + fee
                    #LOSS IS A NEGATIVE NUMBER SO ADD IT TO BALANCE
                    balance += loss
                    totalloss += loss
                    print("Loss short profit: "+str(loss))
                #PROFIT
                else:
                    #PROFIT OF A FULL 1 COIN
                   
                    profit = float(closedata[int(len(closedata) - day - 2)]) - float(lowdata[int(len(lowdata) - day - 1)])
                    percentage = float(profit / closedata[int(len(closedata) - day - 1)]) + 1.0
                    print(percentage)
                    #MULTIPLY PRICE CHANGE PERCENTAGE BY TODAY'S CLOSE
                    profit = float(float(thetradeamount) * percentage) - thetradeamount
                    profit = (profit / 2)
                    profit = profit * leverage
                    profit = profit - slippage  - fee
                    balance += profit
                    totalprofit += profit
                    print("Gain short profit: "+str(profit))
        #plt.plot(balance, label="Balance")
        
        if positiontohold == "Up":
            #plt.scatter(closedata[int(len(closedata)-1)],'g')
            lasttrade = "long"
            tradeamount = float(balance * risk)
        else:
            #plt.scatter(closedata[int(len(closedata)-1)],'r')
            lasttrade = "short"
            tradeamount = float(balance * risk)


        if doubletrade == False:
            printdata2(predicthighdata, highdata, 'high')
            printdata2(predictlowdata, lowdata, 'low')
        else:
            print("No ML data to show, because doubletrade is on")
        print("Balance: "+str(balance))
        
        if profit > loss:
            successful += 1
        else:    
            failed += 1
        perc1 = str(str(successful)+":"+str(failed))
        file = open("currentbalance.txt","w")
        file.write(" Balance: "+str(balance)+" \n Total Loss: "+str(totalloss)+" \n Total Profit: "+str(totalprofit)+" \n Crypto: "+str(crypto))
        file.close()



def realtrade():
    doubletrade()
        

def fullpredict():
    global futuretime
    #futuretime = int(input("Amount of time in future to predict: "))
    futuretime = 1

    predictopendata, opendata = commitpredict(1)
    predicthighdata,highdata = commitpredict(2)
    predictlowdata,lowdata = commitpredict(3)
    predictclosedata,closedata = commitpredict(4)
    predictvolumedata,volumedata = commitpredict(5)

    os.system('cls')

    printdata2(predictopendata,opendata, 'open')
    printdata2(predicthighdata,highdata, 'high')
    printdata2(predictlowdata,lowdata, 'low')
    printdata2(predictvolumedata,volumedata, 'volume')
    printdata3(predictclosedata,closedata,predicthighdata[int(len(predicthighdata) - 1)],predictlowdata[int(len(predictlowdata) - 1)],'close')

        
    plt.plot(predictclosedata, 'bo')
    plt.plot(closedata)
    plt.show()



#backtrade(int(input("Days to backtest: ")),float(input("Risk (amnt to trade): ")), int(input("Leverage: ")))
try:
    realtrade()
except Exception as EEE:
    print("Error: "+str(EEE))
    time.sleep(99999)
