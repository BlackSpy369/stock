import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import time
import yfinance as yf
from utils.utils import preprocess_data

# Page title
st.set_page_config(page_title='Stock Price Predictor', page_icon='📈')
st.title('📈 Stock Price Predictor')

with st.expander('About this app',expanded=False):
    st.markdown('**What can this app do?**')
    st.info('This App can predict Stock Price using ML (Random Forest Regressor and other models)')

    st.markdown('**How to use the app?**')
    st.warning('To engage with the app, go to the sidebar and 1. Select TICKER symbol and n_length and 2. Adjust the model parameters by adjusting the various slider widgets. As a result, this would initiate the ML model building process, display the model results as well as allowing users to download the generated models and accompanying data.')

    st.markdown('Libraries used:')
    st.code('''- Pandas for data wrangling
- Scikit-learn for building a machine learning model
- Altair for chart creation
- Streamlit for user interface
- Yahoo finance for Stock data
  ''', language='markdown')

# Sidebar for accepting input parameters
with st.sidebar:
    # Load data
    st.header('1.1. Input data')

    st.markdown('**1. Use custom data**')
    st.warning("Your File should be a csv file and should contain columns such as Open,Close,Volume etc, else do not use custom data")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, index_col=False)
        

    # Select example data
    st.markdown('**1.2. Use Yahoo Finance data**')
    # ticker=st.selectbox(label="Choose a Stock",options=[" ","META"] ,format_func=lambda x: 'Select an option' if x == '' else x)
    ticker = st.text_input(
        "Enter Yahoo Finance Stock Ticker code", placeholder="e.g. META")
    standarize = st.toggle(label="Standarize Data")
    if ticker:
        data_loader = yf.Ticker(ticker)
        df = pd.DataFrame(data_loader.history(period="max"))

    st.header('2. Set Parameters')
    n_length = st.slider("Select n_length", 10, 100, 30, 5)
    parameter_split_size = st.slider(
        'Data split ratio (% for Training Set)', 10, 90, 80, 5)

    st.subheader('2.1. Learning Parameters')
    with st.expander('See parameters'):
        parameter_n_estimators = st.slider(
            'Number of estimators (n_estimators)', 0, 1000, 100, 100)
        parameter_max_features = st.select_slider(
            'Max features (max_features)', options=['all', 'sqrt', 'log2'])
        parameter_min_samples_split = st.slider(
            'Minimum number of samples required to split an internal node (min_samples_split)', 2, 10, 2, 1)
        parameter_min_samples_leaf = st.slider(
            'Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

    st.subheader('2.2. General Parameters')
    with st.expander('See parameters', expanded=False):
        parameter_random_state = st.slider(
            'Seed number (random_state)', 0, 1000, 42, 1)
        parameter_criterion = st.select_slider('Performance measure (criterion)', options=[
                                               'squared_error', 'absolute_error', 'friedman_mse'])
        parameter_bootstrap = st.select_slider(
            'Bootstrap samples when building trees (bootstrap)', options=[True, False])
        parameter_oob_score = st.select_slider(
            'Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])

    sleep_time = st.slider('Sleep time', 0, 3, 0)

if ticker == "":
    st.warning(
        "Please select a Stock to get Started, either load your own data or use inbuild datasets of stocks.")

if ticker:
    metric = st.selectbox('Select a Metric:', df.columns.insert(
        0, " "), format_func=lambda x: 'Select an option' if x == '' else x)

    if metric == " ":
        st.warning("Select a Metric to get started")

# Initiate the model building process

    if metric != " ":
        if uploaded_file or ticker:
            with st.status("Running ...", expanded=True) as status:
                st.write("Loading data ...")
                time.sleep(sleep_time)

                st.write("Preparing data ...")
                time.sleep(sleep_time)
                X, y = preprocess_data(df[metric].values, n_length,standarize)

                st.write("Splitting data ...")
                time.sleep(sleep_time)

                if standarize:
                    st.write("Standarizing Data...")
                    time.sleep(sleep_time)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(
                    100-parameter_split_size)/100, random_state=parameter_random_state)

                st.write("Model training ...")
                time.sleep(sleep_time)

                if parameter_max_features == 'all':
                    parameter_max_features = None
                    parameter_max_features_metric = X.shape[1]

                rf = RandomForestRegressor(
                    n_estimators=parameter_n_estimators,
                    max_features=parameter_max_features,
                    min_samples_split=parameter_min_samples_split,
                    min_samples_leaf=parameter_min_samples_leaf,
                    random_state=parameter_random_state,
                    criterion=parameter_criterion,
                    bootstrap=parameter_bootstrap,
                    oob_score=parameter_oob_score)
                rf.fit(X_train, y_train)

                st.write("Applying model to make predictions ...")
                time.sleep(sleep_time)
                y_train_pred = rf.predict(X_train)
                y_test_pred = rf.predict(X_test)

                st.write("Evaluating performance metrics ...")
                time.sleep(sleep_time)
                train_mse = mean_squared_error(y_train, y_train_pred)
                train_r2 = r2_score(y_train, y_train_pred)
                test_mse = mean_squared_error(y_test, y_test_pred)
                test_r2 = r2_score(y_test, y_test_pred)

                st.write("Displaying performance metrics ...")
                time.sleep(sleep_time)
                performance_dict = {
                    "Train MSE": train_mse, "Train R2": train_r2, "Test MSE": test_mse, "Test R2": test_r2}
                performance_df = pd.DataFrame(performance_dict, index=range(1))

            status.update(label="Status", state="complete", expanded=True)

            # Display data info
            st.header('Input data', divider='rainbow')
            col = st.columns(4)
            col[0].metric(label="No. of samples", value=X.shape[0], delta="")
            col[1].metric(label="No. of X variables",
                          value=X.shape[1], delta="")
            col[2].metric(label="No. of Training samples",
                          value=X_train.shape[0], delta="")
            col[3].metric(label="No. of Test samples",
                          value=X_test.shape[0], delta="")

            with st.expander('Initial dataset', expanded=True):
                st.dataframe(df, height=210, use_container_width=True)
            with st.expander('Train split', expanded=False):
                train_col = st.columns((3, 1))
                with train_col[0]:
                    st.markdown('**X**')
                    st.dataframe(X_train, height=210,
                                 hide_index=True, use_container_width=True)
                with train_col[1]:
                    st.markdown('**y**')
                    st.dataframe(y_train, height=210,
                                 hide_index=True, use_container_width=True)
            with st.expander('Test split', expanded=False):
                test_col = st.columns((3, 1))
                with test_col[0]:
                    st.markdown('**X**')
                    st.dataframe(X_test, height=210,
                                 hide_index=True, use_container_width=True)
                with test_col[1]:
                    st.markdown('**y**')
                    st.dataframe(y_test, height=210,
                                 hide_index=True, use_container_width=True)

            # Display model parameters
            st.header('Model parameters', divider='rainbow')
            parameters_col = st.columns(3)
            parameters_col[0].metric(
                label="Data split ratio (% for Training Set)", value=parameter_split_size, delta="")
            parameters_col[1].metric(
                label="Number of estimators (n_estimators)", value=parameter_n_estimators, delta="")
            parameters_col[2].metric(
                label="Max features (max_features)", value=parameter_max_features_metric, delta="")

            # Prediction results
            st.header('Prediction results', divider='rainbow')
            s_y_train = pd.Series(
                y_train, name='actual').reset_index(drop=True)
            s_y_train_pred = pd.Series(
                y_train_pred, name='predicted').reset_index(drop=True)
            df_train = pd.DataFrame(
                data=[s_y_train, s_y_train_pred], index=None).T
            df_train['class'] = 'train'

            s_y_test = pd.Series(y_test, name='actual').reset_index(drop=True)
            s_y_test_pred = pd.Series(
                y_test_pred, name='predicted').reset_index(drop=True)
            df_test = pd.DataFrame(
                data=[s_y_test, s_y_test_pred], index=None).T
            df_test['class'] = 'test'

            df_prediction = pd.concat([df_train, df_test], axis=0)

            prediction_col = st.columns((2, 0.2, 3))

            # Display dataframe
            with prediction_col[0]:
                st.dataframe(df_prediction, height=320,
                             use_container_width=True)

            # Display scatter plot of actual vs predicted values
            with prediction_col[2]:
                scatter = alt.Chart(df_prediction).mark_circle(size=60).encode(
                    x='actual',
                    y='predicted',
                    color='class'
                )
                st.altair_chart(scatter, theme='streamlit',
                                use_container_width=True)

            #Displaying Performance Table
            st.table(performance_df)
            st.info("Your Model's **Performance** can be increase or decrease by tuning ***hyperparameters*** (see **sidebar**).")
# Ask for CSV upload if none is detected
else:
    st.warning(
        '👈 Upload a CSV file or click *"Load example data"* to get started!')
