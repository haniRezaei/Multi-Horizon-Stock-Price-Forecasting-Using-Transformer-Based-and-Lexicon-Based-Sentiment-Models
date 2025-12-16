# Multi-Horizon-Stock-Price-Forecasting-Using-Transformer-Based-and-Lexicon-Based-Sentiment-Models
Developed a stock market prediction framework combining financial news sentiment, technical indicators, and historical data to forecast short- and medium-term DJIA movements. Evaluated transformer-based and lexicon-based models, showing that FinBERT improves predictive accuracy for actionable trading insights.
Project Overview

This project investigates the forecasting of stock market movements through the integration of financial news sentiment, technical indicators, and historical stock data. The primary objective was to evaluate the effectiveness of different sentiment analysis methodologies in predicting the directional movement of the Dow Jones Industrial Average (DJIA) across short-term (1-day) and medium-term (5-day) horizons. By combining multiple sources of information, the study aimed to develop a predictive framework capable of providing actionable insights for traders, financial analysts, and investment institutions.

Data and Feature Engineering

The analysis utilized the Daily News for Stock Market Prediction dataset, which merges daily DJIA stock prices spanning 2008 to 2016 with the top 25 financial news headlines per day sourced from Reddit WorldNews. The textual data underwent standard natural language processing (NLP) preprocessing, including tokenization, lemmatization, and the removal of stopwords, to ensure uniformity and facilitate computational analysis.

In addition to sentiment features, a set of technical indicators was engineered to capture the underlying dynamics of the stock market. These included daily percentage returns over 1-, 5-, and 10-day periods, rolling volatility measures, the Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), and day-of-week encoding. The combination of textual sentiment, technical indicators, and historical price data enabled the models to incorporate both qualitative and quantitative dimensions of market behavior.

Sentiment Analysis Methods

Three distinct sentiment analysis techniques were assessed. TextBlob and VADER are lexicon-based methods that quantify sentiment using polarity, subjectivity, or compound scores derived from predefined lexicons. In contrast, FinBERT is a transformer-based language model pre-trained on financial corpora, capable of capturing contextually rich and nuanced sentiment information. The comparative evaluation of these approaches facilitated an assessment of the incremental predictive value provided by transformer-based models relative to traditional lexicon-based methods in financial applications.

Model Development and Evaluation

Six supervised classification algorithms were employed to model directional market movement: Logistic Regression, Linear Discriminant Analysis (LDA), Support Vector Machines (SVM), Random Forest, K-Nearest Neighbors (KNN), and Naive Bayes. A time-series aware cross-validation approach was implemented to prevent information leakage, ensuring that validation datasets temporally followed the corresponding training datasets. Balanced accuracy was utilized as the primary evaluation metric to account for the nearly equal distribution of upward and downward market movements.

Separate models were trained for 1-day and 5-day horizons to enable a rigorous comparison of predictive performance across short-term and medium-term forecasts. All features were lagged by one day to guarantee that the models relied exclusively on historical information, thereby avoiding forward-looking bias.

Results and Insights

Empirical results indicated that FinBERT consistently outperformed lexicon-based sentiment methods across both forecasting horizons. For the 1-day horizon, the KNN classifier combined with FinBERT achieved the highest predictive accuracy, suggesting applicability for short-term trading and intraday decision-making. For the 5-day horizon, SVM and Random Forest models paired with FinBERT demonstrated superior performance, highlighting the effectiveness of contextual sentiment in capturing medium-term market trends. Lexicon-based approaches, such as VADER, provided moderate performance gains when used in combination with ensemble classifiers, suggesting their utility as complementary methods.

These findings underscore the importance of integrating textual sentiment with technical indicators and historical stock data to enhance predictive performance. The results further suggest that medium-term forecasts are more reliable than short-term predictions, providing financial analysts and portfolio managers with robust signals for decision-making that mitigate the influence of daily market noise.

Skills and Applications

This project exemplifies expertise in natural language processing, sentiment analysis, feature engineering for financial time series, supervised machine learning, and time-series forecasting. It demonstrates the practical application of transformer-based models in financial analytics and provides a replicable framework for quantitative finance, algorithmic trading, portfolio management, and market research. By combining textual and quantitative data, the project highlights how advanced analytics can inform trading strategies, risk assessment, and data-driven decision-making in professional financial environments.


Regression-Based Stock Price Prediction

In addition to classification tasks, this project explored regression-based forecasting of the Dow Jones Industrial Average (DJIA) adjusted closing prices using LSTM networks. The modeling approach leveraged the same feature set for all sentiment methods, including sentiment scores, historical stock prices (Open, High, Low, Close, Volume), technical indicators (percentage returns, volatility, RSI, MACD, MACD signal and histogram), and day-of-week encoding. The only difference across experiments was the sentiment input, which consisted of either FinBERT, VADER, or TextBlob scores.

The data was split into a training set (all dates before 2015) and a test set (2015 onwards), with Min-Max scaling applied to both features and target values. Sequences of 30 consecutive trading days were created as input to the LSTM model, allowing the network to capture temporal dependencies in both stock prices and sentiment signals. The LSTM architecture included three stacked layers (128, 64, 32 units) with dropout regularization to prevent overfitting, followed by a dense output layer. The model was trained for up to 200 epochs with early stopping to optimize generalization performance.

Evaluation of the models was conducted using multiple regression metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), R² score, and Mean Absolute Percentage Error (MAPE). The results are summarized as follows:

FinBERT: MSE = 45850.84, RMSE = 214.13, MAE = 165.72, R² = 0.8912, MAPE = 0.96%

VADER: MSE = 50216.96, RMSE = 224.09, MAE = 179.17, R² = 0.8807, MAPE = 1.04%

TextBlob: MSE = 61903.51, RMSE = 248.80, MAE = 200.31, R² = 0.8529, MAPE = 1.16%

These results indicate that FinBERT provides the most accurate predictions in the regression setting, achieving the lowest errors (MSE, RMSE, MAE, MAPE) and the highest R² score. VADER performed moderately well, while TextBlob lagged behind, reflecting the limited capacity of traditional lexicon-based methods to capture nuanced and contextually rich sentiment information. Since the model architecture, training procedure, and features were identical across experiments, the differences in performance can be attributed primarily to the quality and informativeness of the sentiment input.

From an industry perspective, these findings suggest that incorporating transformer-based sentiment, such as FinBERT, into regression models can enhance the prediction of future stock prices, providing more reliable forecasts for traders, portfolio managers, and financial analysts. The results also highlight that while lexicon-based methods can serve as complementary sentiment indicators, transformer-based approaches offer a significant advantage in capturing the complex relationships between market sentiment and price movements.



