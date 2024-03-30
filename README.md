 Sentiment_Analysis
The objective of this Flask web application appears to be sentiment analysis of product reviews. The application allows users to input either a URL of a product page on Flipkart or direct text input.

For URL input:

It scrapes the reviews from the provided Flipkart URL.
It then preprocesses the reviews, extracts the product name from the URL, and predicts the sentiment (positive or negative) of the reviews.
Finally, it counts the number of positive and negative reviews and displays the result along with the product name.

For text input:

It takes the input text, preprocesses it, and predicts the sentiment (positive or negative).
It displays the result (positive or negative).
The application uses a Multinomial Naive Bayes classifier t![30 03 2024_23 17 22_REC](https://github.com/gdhivya2302/Sentiment_Analysis/assets/92502553/ac4cefc4-7059-470c-b424-4793ca854a55)
rained on a dataset of reviews to predict the sentiment of the input reviews or text. Additionally, it utilizes BeautifulSoup for web scraping, NLTK for text preprocessing, and Flask for creating the web application.![30 03 2024_23 17 22_REC](https://github.com/gdhivya2302/Sentiment_Analysis/assets/92502553/4c9a578a-a450-4bea-b31f-e0bd7cdb44da)
![30 03 2024_23 17 57_REC](https://github.com/gdhivya2302/Sentiment_Analysis/assets/92502553/81962bba-cd98-4b32-8c54-98952a889ca2)
![30 03 2024_23 18 23_REC](https://github.com/gdhivya2302/Sentiment_Analysis/assets/92502553/16d9a466-f2c4-4680-af23-bebd8fbb9492)
