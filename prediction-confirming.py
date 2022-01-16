from googleapiclient import model
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from joblib.logger import PrintTime
from pandas.core.algorithms import mode
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from itertools import combinations
import numpy as np
import pandas as pd
from twilio.rest import Client
from toolz.itertoolz import count

client = Client("", "")

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SERVICE_ACCOUNT_FILE = 'pancakeswap-prediction-google-sheets-api-key.json'


credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)


SHEETID= "1yoM9Pu7kAkRTxALzq7k3-VL7Rc6nAJxiED0E7J-H3lE"


service = build('sheets','v4', credentials=credentials)

filename_2 ='tradingview_prediction.csv'
f = open(filename_2, "w+")
f.close() 

 #Call Sheets API
sheet= service.spreadsheets()
#result= sheet.values().get(spreadsheetId=SHEETID,range = "tradingview_prediction_before_10!A1:CP258").execute()

#l_dataset=pd.DataFrame(result.get('values',[]))
#Y_difference=l_dataset[0].values
#X_l1 = l_dataset[1].values
#lnew_dataset=pd.DataFrame(result_ne#.get('values',[]))
#Xnew_l1 = lnew_dataset[0].values
#x= np.array(X_l1).reshape((-1,1))
#y= np.array(Y_difference)
#x_new = np.array(Xnew_l1).reshape((-1,1))
#model=LinearRegression()
#model.fit(x,y)
#r_sq = model.score(x,y)
#print(r_sq)
#print(model.coef_)
#print(x_new)
#print(model.predict(x_new))
def evaluate_model(model, x_test, y_test,sets):
    from sklearn import metrics

    # Predict Test Data 
    y_pred = model.predict(x_test)
    #print(y_pred)
    #print(y_test)
    
    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(y_test, y_pred)
    
    if acc > .55:
     print(str(acc))
     with open('tradingview_prediction.csv','a') as fd:
      fd.write(str(sets)) 
      fd.write(str(acc))
#    if acc>.8:
#      client.messages.create(to="+19542995344", 
 #                      from_="+13192718563", 
#                       body=str(sets) +" "+ str(acc))
#    df_model2 = pd.DataFrame(y_pred)
#    df_model2['YTest'] = list(y_test)
#    df_model2['Results'] = df_model2[0].__eq__(df_model2['YTest'])
    #print(df_model2)

 #   X_2train, X_2test, y_2train, y_2test = train_test_split(x_test,df_model2['Results'],shuffle=True,test_size=0.2,random_state=1)
 #   dtc2 = tree.DecisionTreeClassifier(random_state=0)
    #print(X_train)
    #print(y_train)  
 #   dtc2.fit(X_2train,y_2train)
 #   y2_pred = dtc2.predict(X_2test)
 #   acc2 = metrics.accuracy_score(y_2test,y2_pred)
  #  print("The second model accuracy is" +" " + str(acc2))
 #   if acc2 > .8:
  #   with open('tradingview_prediction.csv','a') as fd:
  ##    fd.write(str(acc2))

   #result = sheet.values().append(spreadsheetId=SHEETID,range = 'acc!A1', valueInputOption="USER_ENTERED", body={"majorDimension":"ROWS","values":acc}).execute()
 
  #  prec = metrics.precision_score(y_test, y_pred)
  #  rec = metrics.recall_score(y_test, y_pred)
  #  f1 = metrics.f1_score(y_test, y_pred)
  #  kappa = metrics.cohen_kappa_score(y_test, y_pred)

    # Calculate area under curve (AUC)
   # y_pred_proba = model.predict_proba(x_test)[::,1]
   # fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
   # auc = metrics.roc_auc_score(y_test, y_pred_proba)

    # Display confussion matrix
   # cm = metrics.confusion_matrix(y_test, y_pred)

   # return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'kappa': kappa, 
   #        'fpr': fpr, 'tpr': tpr, 'auc': auc, 'cm': cm}

def tradingview_classification():
 current_data= sheet.values().get(spreadsheetId=SHEETID,range = "trading_prediction_3!A1:A1000000").execute()
 df_result_ =pd.DataFrame(current_data.get('values',[]))
 row_count= count(df_result_)
 result= sheet.values().get(spreadsheetId=SHEETID,range = 'trading_prediction_3!A2:CP'+str(row_count-2)).execute()
 df_result =pd.DataFrame(result.get('values',[]))
# print(df_result)
 df_result = df_result.drop([0])
 target = df_result[1]
# df_result = df_result[9,25,53]
 df_result= df_result.drop(columns=[0,1,2])
# print(df_result)
 for column in range(1,4):
   for combo in combinations(range(3,94),column):
    features = df_result[list(combo)]
   # print(features)
    if len(combo)==1:
      features=np.array(features)
      features =features.reshape(-1,1)
    else: 
      features = df_result[list(combo)]
    X_train, X_test, y_train, y_test = train_test_split(features,target,shuffle=True,test_size=0.3,random_state=1)
    dtc = tree.DecisionTreeClassifier(random_state=0)
    #print(X_train)
    #print(y_train)  
    dtc.fit(X_train,y_train)
    dtc_eval = evaluate_model(dtc,X_test,y_test,combo)
# print('Accuracy:', dtc_eval['acc'])
# print('Precision:', dtc_eval['prec'])
# print('Recall:', dtc_eval['rec'])
# print('F1 Score:', dtc_eval['f1'])
# print('Cohens Kappa Score:', dtc_eval['kappa'])
# print('Area Under Curve:', dtc_eval['auc'])
# print('Confusion Matrix:\n', dtc_eval['cm']) 

tradingview_classification()


