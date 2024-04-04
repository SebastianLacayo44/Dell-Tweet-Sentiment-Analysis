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

def print_wide(df):
    pd.set_option('display.max_columns',None)
    print(df)
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
# Use adjusted datetime to create 'date' column for grouping purposes
data['date'] = pd.to_datetime(data['datetime_adj']).dt.date


### Combines Saturday / Sunday / Monday data into a single 'day'
# Break datetime into year/month/week/weekday components
data[['year', 'month', 'week', 'weekday']] = data.apply(lambda d: date_parts(d['datetime_adj']), axis=1,
                                                        result_type='expand')
# create list of weekday dates tied to combinations of year/week/weekday to later re-label the combined SatSunMon data with the Monday's date
weekday_key = data[(data['weekday'] != 5) & (data['weekday'] != 6)][
    ['year', 'week', 'weekday', 'date']].drop_duplicates()
# after creating list of final dates, renumber saturday and sunday as 0. When grouping on weekday, this will combine data for those three days
data['weekday'] = data.apply(lambda d: sat_sun_mon_combine(d['weekday']), axis=1)



### Group data by day (Saturday, Sunday, and Monday data grouped together)
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
dd_new_col_order = ['date','year', 'week', 'weekday', 'sentiment', 'MeanSentScore', 'SumSentScore', 'NumTweets']
daily_data = daily_data[dd_new_col_order]
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

### re-order data so that a row represents all data for a single date
def create_pnn_columns(data,time_period_col):
    pos_data = data[data['sentiment']=='positive'][[time_period_col,'MeanSentScore','SumSentScore','NumTweets']]
    pos_data.rename(columns={'MeanSentScore': 'pos.meanSS','SumSentScore': 'pos.sumSS','NumTweets': 'pos.numTweets'},inplace=True)
    neg_data = data[data['sentiment'] == 'negative'][[time_period_col, 'MeanSentScore', 'SumSentScore', 'NumTweets']]
    neg_data.rename(columns={'MeanSentScore': 'neg.meanSS', 'SumSentScore': 'neg.sumSS', 'NumTweets': 'neg.numTweets'},
                    inplace=True)
    neu_data = data[data['sentiment'] == 'neutral'][[time_period_col, 'MeanSentScore', 'SumSentScore', 'NumTweets']]
    neu_data.rename(columns={'MeanSentScore': 'neu.meanSS', 'SumSentScore': 'neu.sumSS', 'NumTweets': 'neu.numTweets'},
                    inplace=True)
    spread_df = pos_data.merge(neg_data,how='outer',on=time_period_col)
    spread_df = spread_df.merge(neu_data,how='outer',on=time_period_col)
    return spread_df

# pass the dataframe and the relevant time period that will be used to join the pos/neg/neu data together
daily_data = create_pnn_columns(daily_data,'stock_date')
data_week = create_pnn_columns(data_week,'week')
data_month = create_pnn_columns(data_month,'month')


### notes on sentiment score metrics
# metrics to experiment with?
# overall sentiment is iffy. The existence of neutral sentiment kind of invalidates this approach, imo
# I think a % of total tweets would be a better way to look at this
# or maybe a 'strongest opinion' metric, where we look at the emotion with the highest sentiment? ex negative tweets had on avg more extreme sentiment that pos tweets


### calculate some sentiment comparison metrics
def sentiment_comparison_metrics(data):
    # % of tweets for each sentiment category
    data['percent_pos'] = data['pos.numTweets'] / (data['pos.numTweets']+data['neg.numTweets']+data['neu.numTweets'])
    data['percent_neg'] = data['neg.numTweets'] / (data['pos.numTweets']+data['neg.numTweets']+data['neu.numTweets'])
    data['percent_neu'] = data['neu.numTweets'] / (data['pos.numTweets']+data['neg.numTweets']+data['neu.numTweets'])
    # sentiment category with the highest avg sentiment score
    data['most_extreme_sentiment'] = data[['pos.meanSS','neg.meanSS','neu.meanSS']].idxmax(axis=1)
    return data

daily_data = sentiment_comparison_metrics(daily_data)
data_week = sentiment_comparison_metrics(data_week)
data_month = sentiment_comparison_metrics(data_month)

# create .csv files with summary statistics
daily_data.to_csv('daily_data_v2.csv',index=False)
data_week.to_csv('weekly_data_v2.csv', index=False)
data_month.to_csv('monthly_data_v2.csv',index=False)




### To-Do -- experiment with how to incorporate emotions

