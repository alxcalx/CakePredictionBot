import re
from web3 import Web3
from web3.types import SignedTx
from cake_contract import contract_address, abi_
import predict as pd
from datetime import datetime as dt
import time, sched

PRIVATE_KEY =''
ADDRESS =''
bsc = 'https://bsc-dataseed.binance.org'
web3 = Web3(Web3.HTTPProvider(bsc))
# TX SETTING
GAS = 400000
GAS_PRICE = 7000000000
# SECONDS LEFT BET AT
SECONDS_LEFT = 5
print(web3.isConnected())

#bnb address
balance = web3.eth.get_balance(ADDRESS)
#bnb balance
balance= web3.fromWei(balance,"ether")
print(balance)


average_winning_odd = .95
accuracy = .6
p_value = accuracy - ((1-accuracy))*(1/average_winning_odd)


next_round =0

    
#pancakeswap contract address
predictionContract = web3.eth.contract(address=contract_address, abi=abi_)

def betBull(value, round):
    bull_bet = predictionContract.functions.betBull(round).buildTransaction({
        'from': ADDRESS,
        'nonce': web3.eth.getTransactionCount(ADDRESS),
        'value': value,
        'gas': GAS,
        'gasPrice': GAS_PRICE,
    })
    signed_tx = web3.eth.account.signTransaction(bull_bet, private_key=PRIVATE_KEY)
    web3.eth.sendRawTransaction(signed_tx.rawTransaction)     
    print(f'{web3.eth.waitForTransactionReceipt(signed_tx.hash)}')

def betBear(value, round):
    bear_bet = predictionContract.functions.betBear(round).buildTransaction({
        'from': ADDRESS,
        'nonce': web3.eth.getTransactionCount(ADDRESS),
        'value': value,
        'gas': GAS,
        'gasPrice': GAS_PRICE,
    })
    signed_tx = web3.eth.account.signTransaction(bear_bet, private_key=PRIVATE_KEY)
    web3.eth.sendRawTransaction(signed_tx.rawTransaction)
    print(f'{web3.eth.waitForTransactionReceipt(signed_tx.hash)}')


def claimReward(round):
    claim_reward = predictionContract.functions.claim(round).buildTransaction({
        'from': ADDRESS,
        'nonce': web3.eth.getTransactionCount(ADDRESS),
        'gas': GAS,
        'gasPrice': GAS_PRICE,
    })
    signed_tx = web3.eth.account.signTransaction(claim_reward, private_key=PRIVATE_KEY)
    web3.eth.sendRawTransaction(signed_tx.rawTransaction)
    print(f'{web3.eth.waitForTransactionReceipt(signed_tx.hash)}')

def checkClaim():
 claimable_rounds= []
 amountofroundsplayed = predictionContract.functions.getUserRoundsLength(ADDRESS).call()
 roundsplayed= predictionContract.functions.getUserRounds(ADDRESS,0,amountofroundsplayed).call()
 for round in roundsplayed[0]:
  if predictionContract.functions.claimable(round,ADDRESS).call() == True:
    claimable_rounds.append(round)
 return claimable_rounds    

def makeBet(epoch):
   result = pd.predict()
   balance = web3.eth.get_balance(ADDRESS)
   balance= web3.fromWei(balance,"ether")
   value_ = p_value*float(balance)
   value = web3.toWei(value_,'ether')
  # current_round=predictionContract.functions.rounds(next_round).call()
  # bear_payout = (current_round[8]/current_round[10])*.97
  # bull_payout = (current_round[8]/current_round[9])*.97
 
   if result == 1 :

      betBull(value, epoch)
      time.sleep(30)

      
      bot()
   if result == -1 :
      
      betBear(value, epoch)
      time.sleep(30)
    
      bot()
   if result ==0:

      time.sleep(30)
      if len(checkClaim()) >0:
       claimReward(checkClaim())
       time.sleep(30)
      
      bot()



def run():
  
    try:
        makeBet(next_round)
               
    except Exception as e:
          print(f'(error) Restarting...% {e}')
           



def bot():
  scheduler = sched.scheduler(time.time, time.sleep)
  global next_round 
  next_round = predictionContract.functions.currentEpoch().call()
  current_round=predictionContract.functions.rounds(next_round).call()
  start_time=current_round[1]
  print(dt.fromtimestamp(start_time))
  print(next_round)
  scheduler.enterabs(start_time+290, 1, run)
  scheduler.run()



if __name__ == '__main__':
    bot()






