import datetime
import numpy as np
import pandas as pd

def print_full(df):
    pd.set_option('display.max_colwidth',None)
    pd.set_option('display.max_rows',None)
    pd.set_option('display.max_columns',None)
    print(df)
    pd.reset_option('display.max_colwidth')
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
def date_parts(datetime):
    weekday = datetime.weekday()
    if weekday in [5,6]:
        # Sat and Sun from week 1 are part of the same 'weekend' as Mon from week 2
        week = datetime.week +1
    else:
        week = datetime.week
    # now that week has been assigned, combine sat/sun/mon into the same day
    month = datetime.month
    year = datetime.year

    return year, month, week, weekday
def sat_sun_mon_combine(daynum):
    if daynum in [1,2,3,4]:
        return daynum
    else:
        return 0

### Get Data
headers=['index','datetime','tweet_ID','text','username','sentiment','sentiment_score','emotion','emotion_score']
data = pd.read_csv('sentiment-emotion-labelled_Dell_tweets.csv', header=0, names=headers)

### Time Notes
# twitter data has datetime in UTC
# google finance data also in UTC, timestamps are 4PM UTC

### Date Adjustment - centering a 'day' of tweets around the stock market closing time
# Ex: stock market closes at 4PM. A tweet at 4:15PM Monday would not affect the Monday closing price, but the Tues price
# subtract 8 hours from every tweet to sync it with the date of the stock closing price it would affect
offset = datetime.timedelta(hours=8)
data['datetime_adj'] = [datetime.datetime.strptime(dt,'%Y-%m-%d %H:%M:%S+00:00')  - offset for dt in data['datetime']]

### Create 'date' column for grouping purposes
data['date'] = pd.to_datetime(data['datetime_adj']).dt.date

### Saturday / Sunday / Monday combination
#Create various labels to allow combination of a week's Saturday, Sunday, and Monday
data[['year','month','week','weekday']] = data.apply(lambda d: date_parts(d['datetime_adj']),axis=1,result_type='expand')
#create list of weekday dates to later re-label the post-SaSuMo combination data with dates
weekday_key = data[(data['weekday'] != 5) & (data['weekday']!= 6)][['year','week','weekday','date']].drop_duplicates()
#after creating list of final dates, renumber saturday and sunday as 0. When grouping on weekday, this will combine data for those three days
data['weekday'] = data.apply(lambda d: sat_sun_mon_combine(d['weekday']), axis=1)

### Group data by day (Saturday, Sunday, and Monday data grouped together
data_day = data[['year','week','weekday','sentiment','sentiment_score']].groupby(['year','week','weekday','sentiment'],as_index=False).agg(
    MeanSentScore = ('sentiment_score', 'mean'),
    SumSentScore = ('sentiment_score', 'sum'),
    NumTweets = ('sentiment_score', np.size))

### divide SumSentScore and NumTweets by 3 to somewhat correct for the combination of Sat/Sun/Mon
data_day['SumSentScore'] = data_day.apply(lambda row: row['SumSentScore']/3 if row['weekday'] == 0 else row['SumSentScore'],axis=1)
data_day['NumTweets'] =data_day.apply(lambda row: row['NumTweets']/3 if row['weekday'] == 0 else row['NumTweets'],axis=1)

### re-apply dates to the grouped dat, to allow for joining on the stock data
daily_data = pd.merge(data_day,weekday_key,how='left',on=['year','week','weekday'])

#re-ordering columns
dd_new_cols = ['date','year', 'week', 'weekday', 'sentiment', 'MeanSentScore', 'SumSentScore', 'NumTweets']
daily_data = daily_data[dd_new_cols]
#renaming date column to make clear that this is an adjusted date
daily_data.rename(columns={'date': 'stock_date'},inplace=True)

### weekly and monthly grouping.
data_week = data[['year','week','sentiment','sentiment_score']].groupby(['year','week','sentiment'],as_index=False).agg(
    MeanSentScore = ('sentiment_score', 'mean'),
    SumSentScore = ('sentiment_score', 'sum'),
    NumTweets = ('sentiment_score', np.size))

data_month = data[['year','month','sentiment','sentiment_score']].groupby(['year','month','sentiment'],as_index=False).agg(
    MeanSentScore = ('sentiment_score', 'mean'),
    SumSentScore = ('sentiment_score', 'sum'),
    NumTweets = ('sentiment_score', np.size))


daily_data.to_csv('daily_data.csv',index=False)
data_week.to_csv('weekly_data.csv', index=False)
data_month.to_csv('monthly_data.csv',index=False)

### To-Do -- experiment with how to incorporate emotions

# get total count of pos / neg / neutral tweets by group
# get summation of sentiment within each group
# experiment with ways to total that sentiment together