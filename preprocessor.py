import re
import pandas as pd

def preprocess(data):
    # Define the pattern to extract date and time from messages
    pattern = '\d{1,2}/\d{1,2}/\d{2}, \d{1,2}:\d{2}\s?(?:AM|PM)\s?-\s?'

    # Split the data using the pattern to get messages
    messages = re.split(pattern, data)[1:]

    # Find all the dates using the pattern
    dates = re.findall(pattern, data)

    # Create a DataFrame with messages and dates
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    # Convert message_date to datetime format
    df['message_date'] = pd.to_datetime(df['message_date'], format='%m/%d/%y, %I:%M %p - ')

    # Rename message_date column to date
    df.rename(columns={'message_date': 'date'}, inplace=True)

    # Function to split user_message into user_name and message
    def split_user_message(message):
        # Pattern to match user_name followed by ': '
        pattern = '^([\w\s]+?):\s(.*)$'
        match = re.match(pattern, message)

        if match:
            return match.group(1), match.group(2)
        else:
            return 'group_notification', message

    # Apply the function to split user_message into user_name and message
    df[['user_name', 'message']] = df['user_message'].apply(lambda x: pd.Series(split_user_message(x)))

    # Drop the original user_message column
    df.drop(columns=['user_message'], inplace=True)

    # Rename user_name column to user
    df.rename(columns={'user_name': 'user'}, inplace=True)

    # Extract year, month, day, etc. from the date
    df['year'] = df['date'].dt.year
    df['day_name'] = df['date'].dt.day_name()
    df['only_date'] = df['date'].dt.date
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # Create a period column for time intervals
    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period

    return df
