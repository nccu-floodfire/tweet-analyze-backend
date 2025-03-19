import pandas as pd
import matplotlib.pyplot as plt
import re

def statisticCalcu(data):
    
    selec = ["id","created_at","from_user_name","from_user_realname", 'from_user_followercount',"text","retweet_id","retweet_count","lang"]
    data = data[selec]
    # how many tweets per week for english tweets and chinese tweets and japanese tweets
   
    data["created_at"] = pd.to_datetime(data["created_at"])
    data.loc[:,"year"] = data["created_at"].dt.year
    data.loc[:,"month"] = data["created_at"].dt.month
    data_deleted = data.copy()

    table1 = pd.pivot_table(data, values='id', index=['year', 'month'], columns=['lang'], aggfunc='count')
    table1 = table1.fillna(0)
    table1 = table1.reset_index()
    table1 = table1.rename(columns={"en":"English","ja":"Japanese","zh":"Chinese"})   

    data_deleted = data_deleted.drop_duplicates(subset=['retweet_id'], keep='first')
    table2 = pd.pivot_table(data_deleted, values='id', index=['year', 'month'], columns=['lang'], aggfunc='count')
    table2 = table2.fillna(0)
    table2 = table2.reset_index()
    table2 = table2.rename(columns={"en":"English2","ja":"Japanese2","zh":"Chinese2"})

    table = pd.merge(table1,table2,on=["year","month"])
    return table