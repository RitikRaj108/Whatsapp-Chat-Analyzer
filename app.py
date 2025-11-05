# Import necessary libraries
import streamlit as st
import preprocessor
import helper
import seaborn as sns
import matplotlib.pyplot as plt

# Set up the sidebar title for the app
st.sidebar.title("WhatsApp Chat Analyzer")

# File uploader widget in the sidebar
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    # Read the uploaded file
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")

    # Preprocess the data to create a DataFrame
    df = preprocessor.preprocess(data)

    # Display the DataFrame in the main area of the app
    st.dataframe(df)

    # Fetch the unique list of users from the chat
    user_list = df['user'].unique().tolist()
    # Remove 'group_notification' from the list
    user_list.remove('group_notification')
    # Sort the user list and add 'Overall' at the beginning
    user_list.sort()
    user_list.insert(0, "Overall")

    # Dropdown to select a user for analysis
    selected_user = st.sidebar.selectbox("Show analysis of this person:", user_list)

    # Analysis type selection dropdown
    analysis_options = [
        "📊 Basic Statistics",
        "😊 Sentiment Analysis",
        "📈 Activity Timeline",
        "👥 Most Busy Users",
        "☁️ Word Cloud",
        "📝 Most Common Words",
        "😀 Emoji Analysis",
        "🔥 Activity Heatmap",
        "🎯 Complete Analysis"
    ]

    selected_analysis = st.sidebar.selectbox("Select Analysis Type:", analysis_options)

    # Button to trigger the analysis
    if st.sidebar.button("Show Analysis"):

        # Always fetch basic statistics
        num_messages, words, no_media_msg, no_links = helper.fetch_stats(selected_user, df)

        # Basic Statistics
        if selected_analysis == "📊 Basic Statistics" or selected_analysis == "🎯 Complete Analysis":
            st.title("📊 Basic Statistics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Messages", num_messages)

            with col2:
                st.metric("Total Words", words)

            with col3:
                st.metric("Media Shared", no_media_msg)

            with col4:
                st.metric("Links Shared", no_links)

            if selected_analysis != "🎯 Complete Analysis":
                st.markdown("---")

        # Sentiment Analysis
        if selected_analysis == "😊 Sentiment Analysis" or selected_analysis == "🎯 Complete Analysis":
            st.title("😊 Sentiment Analysis")
            sentiment_summary = helper.sentiment_analysis(selected_user, df)

            # Display sentiment percentages with proper spacing
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Positive Messages", f"{sentiment_summary['positive']:.1f}%")

            with col2:
                st.metric("Negative Messages", f"{sentiment_summary['negative']:.1f}%")

            with col3:
                st.metric("Neutral Messages", f"{sentiment_summary['neutral']:.1f}%")

            with col4:
                avg_sentiment = sentiment_summary['avg_sentiment']
                sentiment_label = "😊 Positive" if avg_sentiment > 0.1 else "😐 Neutral" if avg_sentiment > -0.1 else "😔 Negative"
                st.metric("Average Sentiment", sentiment_label, f"{avg_sentiment:.3f}")

            # Sentiment Distribution Chart
            st.subheader("📊 Sentiment Distribution")
            col1, col2 = st.columns(2)

            with col1:
                # Pie chart for sentiment distribution
                fig, ax = plt.subplots(figsize=(8, 6))
                sentiment_counts = [sentiment_summary['positive'], sentiment_summary['negative'],
                                    sentiment_summary['neutral']]
                colors = ['#2E8B57', '#DC143C', '#808080']
                labels = ['Positive', 'Negative', 'Neutral']
                wedges, texts, autotexts = ax.pie(sentiment_counts, labels=labels, colors=colors,
                                                  autopct='%1.1f%%', startangle=90)
                ax.set_title('Sentiment Distribution')
                st.pyplot(fig)

            with col2:
                # Bar chart for sentiment scores
                fig, ax = plt.subplots(figsize=(8, 6))
                categories = ['Positive', 'Negative', 'Neutral', 'Compound']
                scores = [sentiment_summary['avg_positive'], sentiment_summary['avg_negative'],
                          sentiment_summary['avg_neutral'], sentiment_summary['avg_sentiment']]
                colors = ['#2E8B57', '#DC143C', '#808080', '#4169E1']
                bars = ax.bar(categories, scores, color=colors)
                ax.set_title('Average Sentiment Scores')
                ax.set_ylabel('Score')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

            # Sentiment Timeline
            st.subheader("📈 Sentiment Timeline")
            sentiment_timeline = helper.sentiment_timeline(selected_user, df)
            if not sentiment_timeline.empty:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(sentiment_timeline['date'], sentiment_timeline['sentiment_score'],
                        color='blue', alpha=0.7, linewidth=2)
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax.fill_between(sentiment_timeline['date'], sentiment_timeline['sentiment_score'],
                                where=(sentiment_timeline['sentiment_score'] > 0), color='green', alpha=0.3,
                                label='Positive')
                ax.fill_between(sentiment_timeline['date'], sentiment_timeline['sentiment_score'],
                                where=(sentiment_timeline['sentiment_score'] < 0), color='red', alpha=0.3,
                                label='Negative')
                ax.set_title('Sentiment Over Time')
                ax.set_xlabel('Date')
                ax.set_ylabel('Sentiment Score')
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

            # Top Positive and Negative Messages
            st.subheader("🎭 Most Positive and Negative Messages")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🟢 Most Positive Messages")
                positive_messages = helper.get_extreme_sentiment_messages(selected_user, df, sentiment_type='positive',
                                                                          top_n=5)
                for i, (msg, score) in enumerate(positive_messages, 1):
                    with st.expander(f"Message {i} (Score: {score:.3f})"):
                        st.write(msg)

            with col2:
                st.markdown("#### 🔴 Most Negative Messages")
                negative_messages = helper.get_extreme_sentiment_messages(selected_user, df, sentiment_type='negative',
                                                                          top_n=5)
                for i, (msg, score) in enumerate(negative_messages, 1):
                    with st.expander(f"Message {i} (Score: {score:.3f})"):
                        st.write(msg)

            if selected_analysis != "🎯 Complete Analysis":
                st.markdown("---")

        # Activity Timeline
        if selected_analysis == "📈 Activity Timeline" or selected_analysis == "🎯 Complete Analysis":
            st.title("📈 Activity Timeline")

            col1, col2 = st.columns(2)

            with col1:
                # Monthly timeline plot
                st.subheader("📅 Monthly Timeline")
                timeline = helper.monthly_timeline(selected_user, df)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(timeline['time'], timeline['message'], color='orange', marker='o', linewidth=2)
                ax.set_title('Monthly Message Activity')
                ax.set_xlabel('Month-Year')
                ax.set_ylabel('Number of Messages')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

            with col2:
                # Daily timeline plot
                st.subheader("📆 Daily Timeline")
                daily_timeline = helper.daily_timeline(selected_user, df)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='pink', alpha=0.7)
                ax.set_title('Daily Message Activity')
                ax.set_xlabel('Date')
                ax.set_ylabel('Number of Messages')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

            # Daily and Monthly activity maps
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Weekly Activity")
                busy_day = helper.week_activity_map(selected_user, df)
                fig, ax = plt.subplots(figsize=(8, 6))
                bars = ax.bar(busy_day.index, busy_day.values, color='lightblue', edgecolor='navy')
                ax.set_title('Messages by Day of Week')
                ax.set_xlabel('Day of Week')
                ax.set_ylabel('Number of Messages')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

            with col2:
                st.subheader("📊 Monthly Activity")
                busy_month = helper.month_activity_map(selected_user, df)
                fig, ax = plt.subplots(figsize=(8, 6))
                bars = ax.bar(busy_month.index, busy_month.values, color='lightcoral', edgecolor='darkred')
                ax.set_title('Messages by Month')
                ax.set_xlabel('Month')
                ax.set_ylabel('Number of Messages')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

            if selected_analysis != "🎯 Complete Analysis":
                st.markdown("---")

        # Most Busy Users (only for Overall)
        if (
                selected_analysis == "👥 Most Busy Users" or selected_analysis == "🎯 Complete Analysis") and selected_user == 'Overall':
            st.title("👥 Most Busy Users")
            x, new_df = helper.most_busy_users(df)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 User Activity Chart")
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(x.index, x.values, color='green', edgecolor='darkgreen')
                ax.set_title('Messages by User')
                ax.set_xlabel('Users')
                ax.set_ylabel('Number of Messages')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

            with col2:
                st.subheader("📈 User Statistics Table")
                st.dataframe(new_df, use_container_width=True)

            if selected_analysis != "🎯 Complete Analysis":
                st.markdown("---")

        elif selected_analysis == "👥 Most Busy Users" and selected_user != 'Overall':
            st.warning("⚠️ 'Most Busy Users' analysis is only available for 'Overall' selection.")

        # Word Cloud
        if selected_analysis == "☁️ Word Cloud" or selected_analysis == "🎯 Complete Analysis":
            st.title("☁️ Word Cloud")
            df_wc = helper.create_wordcloud(selected_user, df)
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(df_wc, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Most Common Words Cloud', fontsize=16, pad=20)
            plt.tight_layout()
            st.pyplot(fig)

            if selected_analysis != "🎯 Complete Analysis":
                st.markdown("---")

        # Most Common Words
        if selected_analysis == "📝 Most Common Words" or selected_analysis == "🎯 Complete Analysis":
            st.title("📝 Most Common Words")
            most_common_df = helper.most_common_words(selected_user, df)
            if not most_common_df.empty:
                col1, col2 = st.columns([2, 1])

                with col1:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    bars = ax.barh(most_common_df[0], most_common_df[1], color='skyblue', edgecolor='navy')
                    ax.set_title('Top 20 Most Common Words')
                    ax.set_xlabel('Frequency')
                    ax.set_ylabel('Words')
                    plt.tight_layout()
                    st.pyplot(fig)

                with col2:
                    st.subheader("📊 Word Frequency Table")
                    st.dataframe(most_common_df.head(10), use_container_width=True)
            else:
                st.info("No common words found.")

            if selected_analysis != "🎯 Complete Analysis":
                st.markdown("---")

        # Emoji Analysis
        if selected_analysis == "😀 Emoji Analysis" or selected_analysis == "🎯 Complete Analysis":
            st.title("😀 Emoji Analysis")
            emoji_df = helper.emoji_helper(selected_user, df)
            if not emoji_df.empty:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📊 Top Emojis Chart")
                    if len(emoji_df) > 0:
                        top_emojis = emoji_df.head(10)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        bars = ax.bar(range(len(top_emojis)), top_emojis[1], color='gold', edgecolor='orange')
                        ax.set_title('Top 10 Most Used Emojis')
                        ax.set_xlabel('Emoji Rank')
                        ax.set_ylabel('Usage Count')
                        ax.set_xticks(range(len(top_emojis)))
                        ax.set_xticklabels(top_emojis[0], fontsize=16)
                        plt.tight_layout()
                        st.pyplot(fig)

                with col2:
                    st.subheader("😊 Emoji Usage Statistics")
                    st.dataframe(emoji_df.head(15), use_container_width=True)
            else:
                st.info("😔 No emojis found in the selected messages.")

            if selected_analysis != "🎯 Complete Analysis":
                st.markdown("---")

        # Activity Heatmap
        if selected_analysis == "🔥 Activity Heatmap" or selected_analysis == "🎯 Complete Analysis":
            st.title("🔥 Activity Heatmap")
            user_heatmap = helper.activity_heatmap(selected_user, df)
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.heatmap(user_heatmap, annot=True, cmap='YlOrRd', ax=ax, fmt='g')
            ax.set_title('Activity Heatmap: Messages by Day and Hour')
            ax.set_xlabel('Time Period')
            ax.set_ylabel('Day of Week')
            plt.tight_layout()
            st.pyplot(fig)

            if selected_analysis != "🎯 Complete Analysis":
                st.markdown("---")

        # Success message
        st.success(f"✅ Analysis completed for {selected_user}!")

        # Show analysis info
        with st.expander("ℹ️ Analysis Information"):
            st.write(f"**Selected User:** {selected_user}")
            st.write(f"**Analysis Type:** {selected_analysis}")
            st.write(f"**Total Messages Analyzed:** {num_messages}")
            st.write(
                f"**Date Range:** {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
