# Import necessary libraries
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
from urlextract import URLExtract
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

# Initialize URL extractor
extract = URLExtract()
analyzer = SentimentIntensityAnalyzer()
def fetch_stats(selected_user, df):
    # Filter data for the selected user if not 'Overall'
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Calculate the number of messages
    num_messages = df.shape[0]

    # Calculate the number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # Calculate the number of media messages
    no_media_msg = df[df['message'] == '<Media omitted>'].shape[0]

    # Calculate the number of links
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), no_media_msg, len(links)

def most_busy_users(df):
    # Find the top users by message count
    x = df['user'].value_counts().head()

    # Calculate the percentage of messages per user
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(columns={'index': 'name', 'user': 'percent'})
    df = df.rename(columns={'percent': 'user_name', 'count': 'percent'})

    return x, df

def create_wordcloud(selected_user, df):
    # Read stop words for word cloud
    with open('stop_hinglish.txt', 'r') as f:
        stop_words = f.read()

    # Filter data for the selected user if not 'Overall'
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Generate the word cloud
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    df_wc = wc.generate(df['message'].str.cat(sep=" "))

    return df_wc

def most_common_words(selected_user, df):
    # Read stop words for common words analysis
    with open('stop_hinglish.txt', 'r') as f:
        stop_words = f.read()
    stop_words = stop_words.split('\n')

    # Filter data for the selected user if not 'Overall'
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Create a list of words excluding stop words and media messages
    temp = df[df['message'] != '<Media omitted>']
    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    # Get the most common words
    most_common_df = pd.DataFrame(Counter(words).most_common(20))

    return most_common_df

def emoji_helper(selected_user, df):
    # Filter data for the selected user if not 'Overall'
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Count the emojis used
    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    # Create a DataFrame with emoji counts
    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df

def monthly_timeline(selected_user, df):
    # Filter data for the selected user if not 'Overall'
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Group by year and month to create a timeline
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    # Create a new column combining year and month for display
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time'] = time

    return timeline

def daily_timeline(selected_user, df):
    # Filter data for the selected user if not 'Overall'
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Group by date to create a daily timeline
    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user, df):
    # Filter data for the selected user if not 'Overall'
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Group by day name to create a weekly activity map
    return df['day_name'].value_counts()

def month_activity_map(selected_user, df):
    # Filter data for the selected user if not 'Overall'
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Group by month name to create a monthly activity map
    return df['month'].value_counts()

def activity_heatmap(selected_user, df):
    # Filter data for the selected user if not 'Overall'
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Create a heatmap for user activity based on day name and period
    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap


def sentiment_analysis(selected_user, df):
    """
    Perform sentiment analysis on messages using VADER sentiment analyzer
    Returns summary statistics of sentiment scores
    """
    # Filter data for the selected user if not 'Overall'
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Filter out media messages and group notifications
    filtered_df = df[(df['message'] != '<Media omitted>') &
                     (df['user'] != 'group_notification') &
                     (df['message'].str.strip() != '')]

    if filtered_df.empty:
        return {
            'positive': 0, 'negative': 0, 'neutral': 100,
            'avg_positive': 0, 'avg_negative': 0, 'avg_neutral': 1, 'avg_sentiment': 0
        }

    # Calculate sentiment scores for each message
    sentiment_scores = []
    for message in filtered_df['message']:
        scores = analyzer.polarity_scores(message)
        sentiment_scores.append(scores)

    # Convert to DataFrame for easier analysis
    sentiment_df = pd.DataFrame(sentiment_scores)

    # Classify messages based on compound score
    def classify_sentiment(compound_score):
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    sentiment_df['sentiment_class'] = sentiment_df['compound'].apply(classify_sentiment)

    # Calculate percentages
    sentiment_counts = sentiment_df['sentiment_class'].value_counts(normalize=True) * 100

    summary = {
        'positive': sentiment_counts.get('positive', 0),
        'negative': sentiment_counts.get('negative', 0),
        'neutral': sentiment_counts.get('neutral', 0),
        'avg_positive': sentiment_df['pos'].mean(),
        'avg_negative': sentiment_df['neg'].mean(),
        'avg_neutral': sentiment_df['neu'].mean(),
        'avg_sentiment': sentiment_df['compound'].mean()
    }

    return summary


def get_sentiment_dataframe(selected_user, df):
    """
    Get detailed sentiment analysis DataFrame
    """
    # Filter data for the selected user if not 'Overall'
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Filter out media messages and group notifications
    filtered_df = df[(df['message'] != '<Media omitted>') &
                     (df['user'] != 'group_notification') &
                     (df['message'].str.strip() != '')].copy()

    if filtered_df.empty:
        return pd.DataFrame()

    # Calculate sentiment scores
    sentiment_data = []
    for idx, row in filtered_df.iterrows():
        scores = analyzer.polarity_scores(row['message'])
        sentiment_data.append({
            'message': row['message'],
            'user': row['user'],
            'date': row['date'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'compound': scores['compound']
        })

    return pd.DataFrame(sentiment_data)


def sentiment_timeline(selected_user, df):
    """
    Create a timeline of sentiment scores over time
    """
    sentiment_df = get_sentiment_dataframe(selected_user, df)

    if sentiment_df.empty:
        return pd.DataFrame()

    # Group by date and calculate average sentiment
    timeline = sentiment_df.groupby(sentiment_df['date'].dt.date).agg({
        'compound': 'mean'
    }).reset_index()

    timeline.columns = ['date', 'sentiment_score']

    return timeline


def get_extreme_sentiment_messages(selected_user, df, sentiment_type='positive', top_n=5):
    """
    Get messages with extreme sentiment scores (most positive or most negative)
    """
    sentiment_df = get_sentiment_dataframe(selected_user, df)

    if sentiment_df.empty:
        return []

    if sentiment_type == 'positive':
        # Sort by compound score in descending order
        sorted_df = sentiment_df.sort_values('compound', ascending=False)
    else:
        # Sort by compound score in ascending order
        sorted_df = sentiment_df.sort_values('compound', ascending=True)

    # Get top N messages
    top_messages = []
    for idx, row in sorted_df.head(top_n).iterrows():
        # Truncate very long messages
        message = row['message']
        if len(message) > 200:
            message = message[:200] + "..."
        top_messages.append((message, row['compound']))

    return top_messages


def sentiment_by_user(df):
    """
    Get sentiment analysis grouped by user (for Overall analysis)
    """
    # Filter out media messages and group notifications
    filtered_df = df[(df['message'] != '<Media omitted>') &
                     (df['user'] != 'group_notification') &
                     (df['message'].str.strip() != '')]

    if filtered_df.empty:
        return pd.DataFrame()

    user_sentiment = {}

    for user in filtered_df['user'].unique():
        user_messages = filtered_df[filtered_df['user'] == user]

        sentiment_scores = []
        for message in user_messages['message']:
            scores = analyzer.polarity_scores(message)
            sentiment_scores.append(scores['compound'])

        user_sentiment[user] = {
            'avg_sentiment': np.mean(sentiment_scores),
            'message_count': len(sentiment_scores),
            'positive_messages': sum(1 for score in sentiment_scores if score >= 0.05),
            'negative_messages': sum(1 for score in sentiment_scores if score <= -0.05),
            'neutral_messages': sum(1 for score in sentiment_scores if -0.05 < score < 0.05)
        }

    # Convert to DataFrame
    result_df = pd.DataFrame(user_sentiment).T
    result_df = result_df.sort_values('avg_sentiment', ascending=False)

    return result_df
