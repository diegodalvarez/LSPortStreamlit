# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 19:58:00 2023

@author: Diego
"""

import datetime as dt
from turtle import back
import yfinance as yf
import streamlit as st

from LSPort import *

st.set_page_config(
    page_title = "L/S Pair Analyzer & Backtesting",
    layout = "wide")

st.header("L/S Pair Analyzer & Backtesting")
st.write("Made by Diego Alvarez")

# helper functions
def bus_day_subtract(date_input):

    if dt.date.weekday(date_input) == 6: date_out = date_input - dt.timedelta(days = 6)
    if dt.date.weekday(date_input) == 5: date_out = date_input - dt.timedelta(days =  5)
    else: date_out = date_input - dt.timedelta(days = 7)

    return date_out

# cache functions
@st.cache_data
def _yf_finance(ticker, start, end):

    return(yf.download(
        tickers = ticker,
        start = start,
        end = end)
        [["Adj Close"]])

# parametrized function on the call-side so that the data is cached
@st.cache_data
def plot_position_rebalance(
    _ls_port_obj,
    lookback_window: int,
    rebalance_method: str,
    backtest_start_date: dt.date,
    backtest_end_date: dt.date):

    fig_weighted, fig_return = _ls_port_obj._plot_position_rebalance(
        lookback_window = lookback_window,
        rebalance_method = "daily",
        backtest_start_date = backtest_start_date,
        backtest_end_date = backtest_end_date)
    
    return(fig_weighted, fig_return)

bad_tickers = {
    "SPX": "^GSPC",
    "MOVE": "^MOVE",
    "VIX": "^VIX",
    "^DJI": "DJI",
    "^IXIC": "IXIC",
    "^RUT": "RUT"}

col1, col2, col3 = st.columns(3)

with col1: 
    st.subheader("Input Data")

    st.write("All Prices Include Dividends and Stock Splits")

with col2:
    
    today_date = dt.date.today()
    
    start_date = st.date_input(
        label = "Start Date",
        value = dt.date(today_date.year - 5, today_date.month, today_date.day))
    
    end_date = st.date_input(
        label = "End Date",
        value = today_date)
    
with col3: 
    
    run_button = st.radio(
        label = "Select Run to extract data",
        options = ["Stop", "Run"])

col1, col2, col3 = st.columns(3)

with col1 : 
    
    long_leg = st.text_input("Long Leg (Ticker)").upper()
    try: long_leg = bad_tickers[long_leg]
    except: pass

    if run_button == "Run":
    
        try:

            df_long = (_yf_finance(
                ticker = long_leg,
                start = start_date,
                end = end_date).
                rename(columns = {"Adj Close": long_leg}))
            
            st.write(df_long.head(5))
            
        except: 
            st.write("There was a problem collecting the data from Yahoo")
        
with col2: 
    
    short_leg = st.text_input("Short Leg").upper()
    try: short_leg = bad_tickers[short_leg]
    except: pass

    if run_button == "Run":
    
        try:
            
            df_short = (_yf_finance(
                ticker = short_leg,
                start = start_date,
                end = end_date).
                rename(columns = {"Adj Close": short_leg}))
            
            st.write(df_short.head(5))
            
        except: 
            st.write("There was a problem collecting the data from Yahoo")
    
with col3: 
    
    benchmark_leg = st.text_input("Benchmark").upper()
    try: benchmark_leg = bad_tickers[benchmark_leg]
    except: pass

    if run_button == "Run":
    
        try:
            df_benchmark = (_yf_finance(
                ticker = benchmark_leg,
                start = start_date,
                end = end_date).
                rename(columns = {"Adj Close": benchmark_leg}))
            
            st.write(df_benchmark.head(5))
            
        except: 
            st.write("There was a problem collecting the data from Yahoo")

if run_button == "Run":

    df_input_full = (df_long.reset_index().merge(
        df_short.reset_index(),
        how = "outer",
        on = ["Date"]).
        merge(
            df_benchmark.reset_index(),
            how = "outer",
            on = ["Date"]).
        set_index("Date"))
    
    df_input_drop = df_input_full.dropna()
    if len(df_input_drop) != len(df_input_full):
        df_input = df_input_drop
        st.write("{} Days Missing and were dropped".format(
            len(df_input_full) - len(df_input_drop)))
        
    else:
        df_input = df_input_full

    ls_port = LSPort(
        long_position = df_input_full[long_leg],
        short_position = df_input_full[short_leg],
        benchmark = df_input_full[benchmark_leg])
    
    sidebar_options = st.sidebar.selectbox(
        label = "Options",
        options = ["backtest"])
    
    col1, col2, col3 = st.columns(3)
    if sidebar_options == "backtest":

        plotting_options = st.sidebar.selectbox(
            label = "Plotting Option",
            options = ["MatplotLib", "Streamlit"])

        with col1:

            rebalance_period = st.selectbox(
                label = "rebalance",
                options = ["daily"])
            
            lookback_window = st.number_input(
                label = "Rolling Beta Window",
                min_value = 1,
                max_value = 365 * 2,
                value = 120)
            
        with col2:

            min_value = df_benchmark.index.min() + dt.timedelta(days = lookback_window)
            max_value = df_benchmark.index.max()

            backtest_start_date = st.date_input(
                label = "Backtest Start Date",
                value = min_value,
                min_value = min_value,
                max_value = max_value)
    
            backtest_end_date = st.date_input(
                label = "Backtest End Date",
                value = max_value,
                min_value = min_value,
                max_value = max_value)
            
            if backtest_start_date > backtest_end_date: st.write("Start Date Needs to be before End Date")       

        with col3:

            backtest_run_button = st.radio(
                label = "Run Button",
                options = ["Stop", "Run"])
            
        if backtest_run_button == "Run":

            fig_weighted, fig_return = plot_position_rebalance(
                _ls_port_obj = ls_port,
                lookback_window = lookback_window,
                rebalance_method = "daily",
                backtest_start_date = backtest_start_date,
                backtest_end_date = backtest_end_date)
            
            st.pyplot(fig_weighted)
            st.pyplot(fig_return)