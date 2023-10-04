import pulp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
from pandas.tseries.offsets import BDay

from tqdm import tqdm
from matplotlib.patches import Patch

import datetime as dt
from dateutil.relativedelta import relativedelta, MO

import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

from LSPair import *


# pass in values as prices not returns
class LSPort(LSPair):

    def __init__(
        self, 
        long_position: pd.Series, 
        short_position: pd.Series, 
        benchmark: pd.Series) -> None:

        if long_position.mean() < 1 or short_position.mean() < 1 or benchmark.mean() < 1:
            print("[ALERT] Passed values as prices not as returns")

        else:

            self.long_position_prices = long_position
            self.short_position_prices = short_position
            self.benchmark_prices = benchmark

            self.long_position_rtns = self.long_position_prices.pct_change().dropna()
            self.short_position_rtns = self.short_position_prices.pct_change().dropna()
            self.benchmark_rtns = self.benchmark_prices.pct_change().dropna()

        super().__init__(
            long_position = self.long_position_rtns,
            short_position = self.short_position_rtns,
            benchmark = self.benchmark_rtns)

    def _get_raw_betas(
        self,  
        security_rtns: pd.Series,
        benchmark_rtns: pd.Series,
        position: str,
        lookback_window: int, 
        verbose: bool) -> pd.DataFrame:

        df_out = (self._rolling_ols(
            df_endog = security_rtns,
            df_exog = benchmark_rtns,
            lookback_windows = [lookback_window],
            conf_int = 0.05,
            verbose = verbose).
            replace({self.benchmark_name: "beta"}).
            query("parameter == 'beta' & variable == 'value'")
            [["Date", "value"]].
            assign(position = position).
            pivot(index = "Date", columns = "position", values = "value").
            reset_index())
        
        return df_out

    def get_raw_betas(self, lookback_window: int, verbose = False) -> pd.DataFrame:

        long_betas = self._get_raw_betas(
            security_rtns = self.long_position_rtns,
            benchmark_rtns = self.benchmark_rtns,
            position = self.long_name,
            lookback_window = lookback_window,
            verbose = verbose)
        
        short_betas = self._get_raw_betas(
            security_rtns = self.short_position_rtns,
            benchmark_rtns = self.benchmark_rtns,
            position = self.short_name,
            lookback_window = lookback_window,
            verbose = verbose)
        
        df_betas = (long_betas.merge(
            short_betas,
            how = "inner",
            on = "Date").
            set_index("Date"))

        return df_betas

    # paramterized function to get data for both plotting functions
    def _plot_raw_betas(
        self,
        lookback_window: int,
        verbose: bool) -> pd.DataFrame:

        return(
            self.get_raw_betas(
                lookback_window = lookback_window,
                verbose = verbose).
               rename(columns = {
                  self.long_name: "long: {}".format(self.long_name),
                  self.short_name: "short: {}".format(self.short_name)}))

    def plot_raw_betas(
          self, 
          lookback_window: int,
          figsize = (8,6), 
          verbose = False) -> plt.Figure:
          
          fig, axes = plt.subplots(figsize = figsize)
          plot_out = (self._plot_raw_betas(
              lookback_window = lookback_window,
              verbose = verbose))
          
          (plot_out.plot(
              ax = axes,
              ylabel = "Beta",
              color = ["mediumblue", "black"],
              title = "{}d Rolling Beta Benchmark: {} from {} to {}".format(
                  lookback_window,
                  self.benchmark_name,
                  plot_out.index.min().date(),
                  plot_out.index.max().date())))
          
          return fig

    def plot_scatter_betas(
        self,
        lookback_window: int,
        figsize = (8,6),
        verbose = False) -> plt.Figure:

        fig, axes = plt.subplots(figsize = figsize)
        plot_out = (self._plot_raw_betas(
            lookback_window = lookback_window,
            verbose = verbose))
        
        (plot_out.plot(
            ax = axes,
            kind = "scatter",
            x = "long: {}".format(self.long_name),
            y = "short: {}".format(self.short_name),
            alpha = 0.3,
            title = "{}d Rolling Beta Benchmark: {} Scatter Plot from {} to {}".format(
                lookback_window, 
                self.benchmark_name,
                plot_out.index.min().date(),
                plot_out.index.max().date())))
        
        return fig

    def _shift(self, df):

        return(df.assign(
            lag_weight = lambda x: x.weight.shift(1)))

    def minimize_betas(
        self,
        lookback_window: int,
        verbose = False):
      
        raw_betas = (self.get_raw_betas(
            lookback_window = lookback_window,
            verbose = verbose))
        
        position_df = pd.DataFrame({
            "ticker": [self.long_name, self.short_name],
            "position": ["long", "short"]})
        
        direction_df = pd.DataFrame({
            "position": ["long", "short"],
            "direction": [1, -1]})

        raw_betas_longer = (raw_betas.reset_index().melt(
            id_vars = raw_betas.index.name).
            rename(columns = {"position": "ticker"}).
            merge(position_df, how = "inner", on = "ticker").
            merge(direction_df, how = "inner", on = "position").
            rename(columns = {"value": "raw_beta"}).
            assign(directional_beta = lambda x: x.raw_beta * x.direction).
            drop(columns = ["direction"]))
            
        df_directional_beta = (raw_betas_longer[
            [raw_betas.index.name, "position", "directional_beta"]].
            pivot(index = "Date", columns = "position", values = "directional_beta").
            sort_index())

        long_weights, short_weights = [], []
        beta_edge_counter = 0

        if verbose == True:

            for long_beta, short_beta in tqdm(zip(
                df_directional_beta.long, df_directional_beta.short)):

                beta_problem = pulp.LpProblem(
                    name = "beta_problem",
                    sense = pulp.LpMinimize)
                
                long_weight = pulp.LpVariable(
                    name = "long_weight",
                    lowBound = 0.1,
                    upBound = 1)
                
                short_weight = pulp.LpVariable(
                    name = "short_weight",
                    lowBound = 0.1,
                    upBound = 1)
                
                beta_problem += (long_beta * long_weight) + (short_beta * short_weight) == 0
                beta_problem += long_weight + short_weight == 1

                solve = beta_problem.solve()
                add_in_long_weight = pulp.value(long_weight)
                add_in_short_weight = pulp.value(short_weight)

                if add_in_long_weight > 1 or add_in_short_weight > 1 and len(long_weights) > 1:

                    add_in_long_weight = long_weights[len(long_weights) - 1]
                    add_in_short_weight = short_weights[len(short_weights) - 1]
                    beta_edge_counter += 1

                long_weights.append(add_in_long_weight)
                short_weights.append(add_in_short_weight)

            weights_df = (pd.DataFrame({
                "long_weight": long_weights,
                "short_weight": short_weights,
                raw_betas.index.name: df_directional_beta.index}).
                melt(id_vars = raw_betas.index.name).
                assign(position = lambda x: x.variable.str.split("_").str[0]))

            df_merge = (raw_betas_longer.merge(
                weights_df, how = "inner", on = [raw_betas.index.name, "position"]).
                drop(columns = ["variable"]).
                rename(columns = {"value": "weight"}).
                groupby("position", group_keys = False).
                apply(self._shift))
            
        else:

            for long_beta, short_beta in zip(
                df_directional_beta.long, df_directional_beta.short):

                beta_problem = pulp.LpProblem(
                    name = "beta_problem",
                    sense = pulp.LpMinimize)
                
                long_weight = pulp.LpVariable(
                    name = "long_weight",
                    lowBound = 0.1,
                    upBound = 1)
                
                short_weight = pulp.LpVariable(
                    name = "short_weight",
                    lowBound = 0.1,
                    upBound = 1)
                
                beta_problem += (long_beta * long_weight) + (short_beta * short_weight) == 0
                beta_problem += long_weight + short_weight == 1

                solve = beta_problem.solve()
                add_in_long_weight = pulp.value(long_weight)
                add_in_short_weight = pulp.value(short_weight)

                if add_in_long_weight > 1 or add_in_short_weight > 1 and len(long_weights) > 1:

                    add_in_long_weight = long_weights[len(long_weights) - 1]
                    add_in_short_weight = short_weights[len(short_weights) - 1]
                    beta_edge_counter += 1

                long_weights.append(add_in_long_weight)
                short_weights.append(add_in_short_weight)

            weights_df = (pd.DataFrame({
                "long_weight": long_weights,
                "short_weight": short_weights,
                raw_betas.index.name: df_directional_beta.index}).
                melt(id_vars = raw_betas.index.name).
                assign(position = lambda x: x.variable.str.split("_").str[0]))

            df_merge = (raw_betas_longer.merge(
                weights_df, how = "inner", on = [raw_betas.index.name, "position"]).
                drop(columns = ["variable"]).
                rename(columns = {"value": "weight"}).
                groupby("position", group_keys = False).
                apply(self._shift))

        return df_merge

    def plot_weights(
        self,
        lookback_window: int,
        figsize = (16,6),
        verbose = False):
      
        port = self.minimize_betas(
            lookback_window = lookback_window,
            verbose = verbose)

        fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = figsize)
        
        df_weight = (port[
            ["Date", "ticker", "lag_weight"]].
            assign(lag_weight = lambda x: x.lag_weight * 100).
            pivot(index = "Date", columns = "ticker", values = "lag_weight"))
        
        df_weight.plot(
            ax = axes[0],
            title = "Individual Weights",
            ylabel = "Holding (%)",
            color = ["mediumblue", "black"])
        
        df_weight_stacked = (df_weight.assign(
            short_weight = df_weight[self.short_name],
            long_weight = 100).
            drop(columns = [self.short_name, self.long_name]).
            rename(columns = {
                "long_weight": self.long_name,
                "short_weight": self.short_name}))
        
        df_weight_stacked.plot(
            ax = axes[1],
            color = "black",
            legend = False,
            title = "Stacked Weight")
        
        axes[1].fill_between(
            x = df_weight_stacked.index,
            y1 = df_weight_stacked[self.short_name],
            y2 = df_weight_stacked[self.long_name],
            where = df_weight_stacked[self.long_name] > df_weight_stacked[self.short_name],
            facecolor = "Green",
            alpha = 0.3,
            label = self.long_name)
        
        axes[1].fill_between(
            x = df_weight_stacked.index,
            y1 = df_weight_stacked[self.short_name],
            y2 = 0,
            where = df_weight_stacked[self.short_name] > 0,
            facecolor = "Red",
            alpha = 0.3,
            label = self.short_name)
        
        legend_elements = [
            Patch(facecolor = "green", alpha = 0.3, label = self.long_name),
            Patch(facecolor = "red", alpha = 0.3, label = self.short_name)]

        axes[1].legend(handles = legend_elements)
        
        fig.suptitle("Beta Neutral Holding Long: {} Short: {} Benchmark from {} to {}".format(
                self.long_name,
                self.short_name,
                self.benchmark_name,
                df_weight.index.min().date(), 
                df_weight.index.max().date()))
        
        plt.tight_layout()

        return fig

    def get_weighted_beta(
        self,
        lookback_window: int,
        verbose = False) -> pd.DataFrame:
      
        port = self.minimize_betas(
            lookback_window = lookback_window,
            verbose = verbose)
        
        weighted_beta = (port[
            ["Date", "position", "directional_beta", "lag_weight"]].
            assign(weighted_beta = lambda x: x.lag_weight * x.directional_beta).
            drop(columns = ["lag_weight", "directional_beta"]).
            pivot(index = "Date", columns = "position", values = "weighted_beta").
            dropna().
            assign(port_beta = lambda x: x.long + x.short).
            rename(columns = {
                "long": self.long_name,
                "short": self.short_name}))
        
        return weighted_beta

    def plot_weighted_beta(
        self, 
        lookback_window: int,
        figsize = (16, 6),
        verbose = False) -> plt.Figure:
      
        weighted_beta = self.get_weighted_beta(
            lookback_window = lookback_window,
            verbose = verbose)
        
        fig, axes = plt.subplots(ncols = 2, figsize = figsize)

        (weighted_beta.drop(
            columns = ["port_beta", self.short_name]).
            plot(
                ax = axes[0],
                color = "mediumblue",
                ylabel = "Positive Beta"))
        
        axes_copy = axes[0].twinx()
        axes_copy.invert_yaxis()

        (weighted_beta.drop(
            columns = ["port_beta", self.long_name]).
            plot(
                ax = axes_copy,
                color = "black",
                title = "Beta Comparison"))
        axes_copy.set_ylabel("Negative Beta (Inverted)", rotation = 270, labelpad = 15)
        
        axes[0].legend(loc = "upper left")
        axes_copy.legend(loc = "upper right")

        (weighted_beta.drop(
            columns = [self.long_name, self.short_name]).
            plot(
                ax = axes[1],
                ylabel = "Beta",
                color = "mediumblue",
                title = "Portfolio Beta"))
        
        fig.suptitle("Long: {} Short: {} with rolling OLS window of {} from {} to {}".format(
            self.long_name, self.short_name,
            lookback_window,
            weighted_beta.index.min().date(),
            weighted_beta.index.max().date()))
        
        plt.tight_layout()
        return fig

    def _cum_rtn(self, df):
        return(df.assign(
            cum_rtn = lambda x: (np.cumprod(1 + x.weighted_rtn) - 1) * 100))

    def get_position_performance(
        self,
        lookback_window: int,
        verbose = False) -> pd.DataFrame:

        if verbose == True:
            print("[INFO] Minimizing Betas")

        port = self.minimize_betas(
            lookback_window = lookback_window,
            verbose = verbose)
        
        df_rtns = (pd.DataFrame({
            "long": self.long_position_rtns,
            "short": self.short_position_rtns}).
            reset_index().
            melt(id_vars = "Date").
            rename(columns = {
                "variable": "position",
                "value": "rtn"}))
        
        df_weighted_rtn = (port[
            ["Date", "position", "lag_weight"]].
            merge(df_rtns, how = "inner", on = ["position", "Date"]).
            assign(weighted_rtn = lambda x: x.lag_weight * x.rtn)
            [["Date", "position", "weighted_rtn"]].
            groupby("position", group_keys = False).
            apply(self._cum_rtn).
            dropna())
        
        return df_weighted_rtn

    def plot_position_performance(
        self,
        lookback_window: int,
        figsize = (8,6),
        verbose = False) -> plt.Figure:
      
        df_weighted_rtn = (self.get_position_performance(
            lookback_window = lookback_window,
            verbose = verbose)
            [["Date", "position", "cum_rtn"]].
            pivot(index = "Date", columns = "position", values = "cum_rtn"))

        fig, axes = plt.subplots(figsize = figsize)

        (df_weighted_rtn.plot(
            ax = axes,
            ylabel = "Cumulative Return (%)",
            color = ["mediumblue", "black"],
            title = "Long: {} Short: {} Weighted Return from {} to {}".format(
                self.long_name,
                self.short_name,
                df_weighted_rtn.index.min().date(),
                df_weighted_rtn.index.max().date())))

        axes.fill_between(
            x = df_weighted_rtn.index,
            y1 = df_weighted_rtn["long"].values,
            y2 = df_weighted_rtn["short"].values,
            where = df_weighted_rtn["long"].values > df_weighted_rtn["short"].values,
            facecolor = "green",
            alpha = 0.3)
        
        axes.fill_between(
            x = df_weighted_rtn.index,
            y1 = df_weighted_rtn["short"].values,
            y2 = df_weighted_rtn["long"].values,
            where = df_weighted_rtn["short"].values > df_weighted_rtn["long"].values,
            facecolor = "red",
            alpha = 0.3)
        
        return fig

    def get_port_performance(
        self,
        lookback_window: int,
        verbose = False) -> pd.DataFrame:

        df_weighted_rtn = self.get_position_performance(
            lookback_window = lookback_window,
            verbose = verbose)
        
        df_port = (df_weighted_rtn[
            ["Date", "position", "weighted_rtn"]].
            pivot(index = "Date", columns = "position", values = "weighted_rtn").
            assign(
                port_rtn = lambda x: x.long + (-1 * x.short),
                port_cum_rtn = lambda x: (np.cumprod(1 + x.port_rtn) - 1) * 100).
            drop(columns = ["long", "short"]))
        
        return df_port

    def plot_port_performance(
        self,
        lookback_window: int,
        figsize = (12,6),
        verbose = False) -> plt.Figure:

        df_port = self.get_port_performance(
            lookback_window = lookback_window,
            verbose = verbose)
        
        fig, axes = plt.subplots(figsize = figsize)
        (df_port.drop(
            columns = ["port_rtn"]).
            plot(
                ax = axes,
                legend = False,
                ylabel = "Cumulative Return %",
                color = "mediumblue",
                title = "Long: {} Short: {} From {} to {}".format(
                    self.long_name,
                    self.short_name,
                    df_port.index.min().date(),
                    df_port.index.max().date())))
        
        axes.fill_between(
            x = df_port.index,
            y1 = df_port["port_cum_rtn"].values,
            y2 = 0,
            where = df_port["port_cum_rtn"].values > 0,
            facecolor = "green",
            alpha = 0.3)
        
        axes.fill_between(
            x = df_port.index,
            y1 = df_port["port_cum_rtn"].values,
            y2 = 0,
            where = df_port["port_cum_rtn"].values < 0, 
            facecolor = "red",
            alpha = 0.3) 
        
        return fig
    
    def _get_min_date(self, df) -> pd.DataFrame:

        return(df.query("Date == Date.min()"))

    def position_rebalance(
        self,
        lookback_window: int,
        rebalance_method: str,
        verbose = False):
      
        # return 1 dataframe for weighting and single performance and beta
        # return 1 dataframe for portfolio return

        rebalance_methods = ["daily", "weekly", "monthly", "quarterly", "yearly"]

        if rebalance_method not in rebalance_methods:

            print("Rebelance Method must be one of the following", rebalance_method)
            return - 1 

        else:

            full_weighting = self.minimize_betas(
                lookback_window = lookback_window,
                verbose = verbose)
            
            df_returns = (pd.DataFrame({
                self.long_name: self.long_position_rtns,
                self.short_name: self.short_position_rtns}).
                reset_index().
                melt(id_vars = self.long_position_rtns.index.name).
                rename(columns = {
                    "variable": "ticker",
                    "value": "rtns"}))
            
            if rebalance_method == "daily":
            
                df_position_out = (full_weighting.drop(
                    columns = ["weight"]).
                    merge(df_returns, how = "inner",  on = ["Date", "ticker"]).
                    assign(
                        weighted_raw_beta = lambda x: x.raw_beta * x.lag_weight,
                        weighted_directional_beta = lambda x: x.directional_beta * x.lag_weight,
                        weighted_rtn = lambda x: x.rtns * x.lag_weight).
                        groupby("position", group_keys = False).
                        apply(self._cum_rtn))
                
            if rebalance_method == "weekly":

                df_weighting = (full_weighting[[
                    "Date", "ticker", "lag_weight"]].
                    assign(weekday = lambda x: x.Date.dt.dayofweek).
                    query("weekday == 0").
                    drop(columns = ["weekday"]))
                
                df_position_out = (full_weighting[
                    ["Date", "ticker", "raw_beta", "position", "directional_beta"]].
                    merge(df_weighting, how = "left", on = ["Date", "ticker"]).
                    assign(lag_weight = lambda x: x.lag_weight.ffill()).
                    dropna().
                    merge(df_returns, how = "inner", on = ["Date", "ticker"]).
                    assign(
                        weighted_raw_beta = lambda x: x.raw_beta * x.lag_weight,
                        weighted_directional_beta = lambda x: x.directional_beta * x.lag_weight,
                        weighted_rtn = lambda x: x.rtns * x.lag_weight).
                    groupby("position", group_keys = False).
                    apply(self._cum_rtn))
                
            if rebalance_method == "monthly":

                df_weighting = (full_weighting.drop(
                    columns = ["weight"]).
                    assign(month_year = lambda x: x.Date.dt.year.astype(str) + "_" + x.Date.dt.month.astype(str)).
                    groupby("month_year").
                    apply(self._get_min_date).
                    drop(columns = ["month_year"]).
                    reset_index(drop = True))
                
                df_position_out = (full_weighting.drop(
                    columns = ["weight", "lag_weight"]).
                    merge(df_weighting, how = "left", on = ["Date", "ticker", "raw_beta", "position", "directional_beta"]).
                    assign(lag_weight = lambda x: x.lag_weight.ffill()).
                    dropna().
                    merge(df_returns, how = "inner", on = ["Date", "ticker"]).
                    assign(
                        weighted_raw_beta = lambda x: x.lag_weight * x.raw_beta,
                        weighted_directional_beta = lambda x: x.lag_weight * x.raw_beta,
                        weighted_rtn = lambda x: x.rtns * x.lag_weight).
                    groupby("position", group_keys = False).
                    apply(self._cum_rtn))
                
            if rebalance_method == "quarterly":

                df_weighting = (full_weighting.drop(
                    columns = ["weight"]).
                    assign(quarter_year = lambda x: x.Date.dt.year.astype(str) + "_" + x.Date.dt.quarter.astype(str)).
                    groupby("quarter_year").
                    apply(self._get_min_date).
                    drop(columns = ["quarter_year"]).
                    reset_index(drop = True))

                df_position_out = (full_weighting.drop(
                    columns = ["weight", "lag_weight"]).
                    merge(df_weighting, how = "left", on = ["Date", "ticker", "raw_beta", "position", "directional_beta"]).
                    assign(lag_weight = lambda x: x.lag_weight.ffill()).
                    dropna().
                    merge(df_returns, how = "inner", on = ["Date", "ticker"]).
                    assign(
                        weighted_raw_beta = lambda x: x.lag_weight * x.raw_beta,
                        weighted_directional_beta = lambda x: x.lag_weight * x.raw_beta,
                        weighted_rtn = lambda x: x.rtns * x.lag_weight).
                    groupby("position", group_keys = False).
                    apply(self._cum_rtn))
                
            if rebalance_method == "yearly":

                df_weighting = (full_weighting.drop(
                    columns = ["weight"]).
                    assign(year = lambda x: x.Date.dt.year).
                    groupby("year").
                    apply(self._get_min_date).
                    drop(columns = ["year"]).
                    reset_index(drop = True))

                df_position_out = (full_weighting.drop(
                    columns = ["weight", "lag_weight"]).
                    merge(df_weighting, how = "left", on = ["Date", "ticker", "raw_beta", "position", "directional_beta"]).
                    assign(lag_weight = lambda x: x.lag_weight.ffill()).
                    dropna().
                    merge(df_returns, how = "inner", on = ["Date", "ticker"]).
                    assign(
                        weighted_raw_beta = lambda x: x.lag_weight * x.raw_beta,
                        weighted_directional_beta = lambda x: x.lag_weight * x.raw_beta,
                        weighted_rtn = lambda x: x.rtns * x.lag_weight).
                    groupby("position", group_keys = False).
                    apply(self._cum_rtn))

            df_port_out = (df_position_out[
                ["Date", "position", "weighted_rtn"]].
                pivot(index = "Date", columns = "position", values = "weighted_rtn").
                assign(port_rtn = lambda x: x.long + (-1* x.short)).
                dropna().
                assign(cum_port_rtn = lambda x: (np.cumprod(1 + x.port_rtn) - 1) * 100))
                
            return df_position_out, df_port_out
    
    def plot_position_rebalance(
        self,
        lookback_window: int,
        rebalance_method: str,
        figsize = (28, 6),
        verbose = False):
      
        rebalance_methods = ["daily", "weekly", "monthly", "quarterly", "yearly"]

        if rebalance_method not in rebalance_methods:

            print("Rebelance Method must be one of the following", rebalance_method)
            return - 1 

        else:

            df_position, df_port = self.position_rebalance(
                lookback_window = lookback_window,
                rebalance_method = rebalance_method,
                verbose = False)
            
            fig_weighted, axes_weighted = plt.subplots(ncols = 2, figsize = figsize)

            df_stacked = (df_position[
                ["Date", "position", "lag_weight"]].
                assign(lag_weight = lambda x: x.lag_weight * 100).
                pivot(index = "Date", columns = "position", values = "lag_weight").
                dropna().
                assign(long = lambda x: 100))
            
            (df_stacked.plot(
                ax = axes_weighted[0],
                ylabel = "Weight (%)",
                title = "Portfolio Rebalance Schedule",
                color = "black",
                legend = False))
            
            axes_weighted[0].fill_between(
                  x = df_stacked.index,
                  y1 = df_stacked["long"],
                  y2 = df_stacked["short"],
                  where = df_stacked["long"] > df_stacked["short"],
                  facecolor = "Green",
                  alpha = 0.3,
                  label = "long")
              
            axes_weighted[0].fill_between(
                x = df_stacked.index,
                y1 = df_stacked["short"],
                y2 = 0,
                where = df_stacked["short"] > 0,
                facecolor = "Red",
                alpha = 0.3,
                label = "short")
            
            legend_elements = [
                Patch(facecolor = "green", alpha = 0.3, label = "long"),
                Patch(facecolor = "red", alpha = 0.3, label = "short")]

            axes_weighted[0].legend(handles = legend_elements)
                      
            df_weighted_beta = (df_position.assign(
                weighted_beta = lambda x: x.lag_weight * x.directional_beta)
                [["Date", "position", "weighted_beta"]].
                pivot(index = "Date", columns = "position", values = "weighted_beta").
                dropna())
            
            axes_weighted_copy = axes_weighted[1].twinx()
            axes_weighted_copy.invert_yaxis()

            (df_weighted_beta[
                ["long"]].
                plot(
                    ax = axes_weighted[1],
                    ylabel = "Long Beta",
                    color = "mediumblue"))
            
            (df_weighted_beta[
                ["short"]].
                plot(
                    ax = axes_weighted_copy,
                    ylabel = "Short Beta",
                    color = "black",
                    title = "Individual Beta Matching"))

            axes_weighted_copy.set_ylabel("Negative Beta (Inverted)", rotation = 270, labelpad = 15)
            axes_weighted[1].legend(loc = "upper left")
            axes_weighted_copy.legend(loc = "upper right")

            fig_weighted.suptitle("Long: {} Short: {} Benchmark: {} {} rolling OLS rebalance: {} from {} to {}".format(
                self.long_name, self.short_name, self.benchmark_name,
                lookback_window, rebalance_method,
                df_weighted_beta.index.min().date(),
                df_weighted_beta.index.max().date()))
            
            fig_port, axes_port = plt.subplots(ncols = 3, figsize = figsize)

            beta_exposure = (df_position.assign(
                weighted_beta = lambda x: x.directional_beta * x.lag_weight)
                [["Date", "position", "weighted_beta"]].
                pivot(index = "Date", columns = "position", values = "weighted_beta").
                assign(beta_exposure = lambda x: x.long + x.short)
                [["beta_exposure"]].
                dropna().
                rename(columns = {"beta_exposure": "beta exposure"}).
                plot(
                    ax = axes_port[0],
                    ylabel = "Beta",
                    title = "Portfolio Beta Exposure",
                    color = "mediumblue"))
            
            position_cum_rtn = (df_position[
                ["Date", "position", "cum_rtn"]].
                pivot(index = "Date", columns = "position", values = "cum_rtn"))
            
            position_cum_rtn.plot(
                ax = axes_port[1],
                title = "Position Cumulative Return",
                color = ["mediumblue", "black"],
                ylabel = "Cumulative Return (%)")
            
            axes_port[1].fill_between(
                x = position_cum_rtn.index,
                y1 = position_cum_rtn["long"].values,
                y2 = position_cum_rtn["short"].values,
                where = position_cum_rtn["long"].values > position_cum_rtn["short"].values,
                facecolor = "green",
                alpha = 0.3)
            
            axes_port[1].fill_between(
                x = position_cum_rtn.index,
                y1 = position_cum_rtn["short"].values,
                y2 = position_cum_rtn["long"].values, 
                where = position_cum_rtn["long"].values < position_cum_rtn["short"].values,
                facecolor = "red",
                alpha = 0.3)
            
            df_port.plot(
                ax = axes_port[2],
                title = "Portfolio Return",
                color = "black",
                ylabel = "Cumlative Return (%)",
                legend = False)
            
            axes_port[2].fill_between(
                x = df_port.index,
                y1 = df_port["cum_port_rtn"].values,
                y2 = 0,
                where = df_port["cum_port_rtn"].values > 0,
                facecolor = "green",
                alpha = 0.3)
            
            axes_port[2].fill_between(
                x = df_port.index,
                y1 = df_port["cum_port_rtn"].values,
                y2 = 0,
                where = df_port["cum_port_rtn"].values < 0,
                facecolor = "red",
                alpha = 0.3)
            
        fig_port.suptitle("Long: {} Short: {} Benchmark: {} {} Rolling OLS Rebaalnce: {} from {} to {}".format(
            self.long_name, self.short_name, self.benchmark_name,
            lookback_window, rebalance_method,
            df_weighted_beta.index.min().date(),
            df_weighted_beta.index.max().date()))

        plt.tight_layout()

    def _get_next_rebal_date(self, rebalance_method: str) -> dt.datetime:
        
        last_date = self.long_position_rtns.index.max().date()

        if rebalance_method == "daily":
            next_date = last_date + BDay(1)
            next_date = dt.date(year = next_date.year, month = next_date.month, day = next_date.day)

        if rebalance_method == "weekly":
            next_date = last_date + relativedelta(weekday = MO(1))

        if rebalance_method == "monthly":

            if last_date.month == 12: next_month = dt.date(year = last_date.year + 1, month = 1, day = 1)
            else: next_month = dt.date(year = last_date.year, month = last_date.month + 1, day = 1)
            
            next_date = None
            while next_date is None:

                if next_month.weekday() < 5: next_date = next_month
                else: next_month += dt.timedelta(days = 1)

        if rebalance_method == "quarterly":

            quarter_end = last_date + pd.tseries.offsets.QuarterEnd() + dt.timedelta(days = 1)
            next_date = None
            while next_date is None:

                if quarter_end.weekday() < 5: next_date = quarter_end
                else: quarter_end = quarter_end + dt.timedelta(days = 1)

        if rebalance_method == "yearly":

            year_end = last_date + pd.tseries.offsets.YearBegin()
            next_date = None
            while next_date is None:

                if year_end.weekday() < 5: next_date = year_end
                else: year_end = year_end + dt.timedelta(days = 1)

        return last_date, next_date


    def _tomorrow_rebalance(self, rebalance_method: str) -> bool:

        last_date, next_date = self._get_next_rebal_date(
              rebalance_method = rebalance_method)
        
        check_date = last_date + dt.timedelta(days = 1)
        tomorrow_date = None
        while tomorrow_date is None:

            if check_date.weekday() < 5: tomorrow_date = check_date
            else: check_date = check_date + dt.timedelta(days = 1)

        if tomorrow_date == next_date: return True
        else: return False

    def tomorrow_rebalance(
        self, 
        rebalance_method: str,
        lookback_window: int,
        verbose = False) -> pd.DataFrame:

        rebalance = self._tomorrow_rebalance(rebalance_method)

        if rebalance == True:

            raw_betas = (self.get_raw_betas(
                lookback_window = lookback_window,
                verbose = verbose).
                tail(1))
            
            long_beta = raw_betas[self.long_name].iloc[0]
            short_beta = raw_betas[self.short_name].iloc[0] * -1

            beta_problem = pulp.LpProblem(
                    name = "beta_problem",
                    sense = pulp.LpMinimize)
                
            long_weight = pulp.LpVariable(
                name = "long_weight",
                lowBound = 0.1,
                upBound = 1)
            
            short_weight = pulp.LpVariable(
                name = "short_weight",
                lowBound = 0.1,
                upBound = 1)
            
            beta_problem += (long_beta * long_weight) + (short_beta * short_weight) == 0
            beta_problem += long_weight + short_weight == 1

            solve = beta_problem.solve()

            long_weight_output, short_weight_output = pulp.value(long_weight), pulp.value(short_weight)

            df_out = pd.DataFrame(
                index = ["Weighting"],
                data = {
                    "Long Beta": [long_beta],
                    "Short Beta": [short_beta],
                    "Target Long Weighting": [long_weight_output],
                    "Target Short Weighting": [short_weight_output],
                    "Target Beta Exposure": [(long_weight_output * long_beta) + (short_weight_output * short_beta)]})

            return df_out

        else: return None

    # an initializer like function to keep the variable in member
    def _set_initial_capital(self, initial_capital: float): self.port_capital = pd.Series(initial_capital)

    # shift to get shares
    def _shift_shares(self, df):
        return(df.assign(old_shares = lambda x: x.num_shares.shift(1).fillna(0)))

    def _pnl(self, df: pd.DataFrame):

        df_tmp = (df.assign(
            port_value = self.port_capital.iloc[-1],
            allocated_cash = lambda x: x.port_value * x.lag_weight,
            num_shares = lambda x: (x.allocated_cash / x.price).astype(int),
            position_value = lambda x: x.num_shares * x.price,
            cash = lambda x: x.allocated_cash - x.position_value,
            position_new_value = lambda x: ((x.rtns * np.sign(x.num_shares)) + 1) * x.position_value,
            position_pnl = lambda x: x.position_new_value - x.position_value,
            weighted_rtn = lambda x: x.lag_weight * x.rtns * np.sign(x.num_shares)))

        new_port_value = df_tmp.position_new_value.sum() + df_tmp.cash.sum()
        self.port_capital = pd.concat([self.port_capital, pd.Series(new_port_value)])

        return df_tmp

    def _cum_single_values(self, df):

        return(df.sort_values(
            "Date").
            assign(
                cum_pnl = lambda x: np.cumsum(x.position_pnl),
                cum_rtn = lambda x: (np.cumprod(1 + x.weighted_rtn) - 1) * 100))
 
    def get_backtest(
        self,
        lookback_window: int,
        rebalance_method: str,
        initial_capital: float):
      
        self._set_initial_capital(initial_capital)

        prices_df = (pd.DataFrame(
            [self.long_position_prices, -1 * self.short_position_prices]).
            T.
            reset_index().
            melt(id_vars = self.long_position_prices.index.name).
            rename(columns = {
                "variable": "ticker",
                "value": "price"}))

        df_position, df_port = self.position_rebalance(
                lookback_window = lookback_window,
                rebalance_method = rebalance_method,
                verbose = False)
        
        df_position_backtest = (df_position[
            ["Date", "ticker", "position", "directional_beta", "rtns", "lag_weight"]].
            merge(prices_df, how = "left", on = ["Date", "ticker"]).
            dropna().
            sort_values("Date"))
        
        df_purchase_book = (df_position_backtest.groupby(
            "Date", group_keys = False).
            apply(self._pnl).
            groupby("ticker", group_keys = False).
            apply(self._shift_shares).
            assign(change_shares = lambda x: x.num_shares - x.old_shares).
            groupby("ticker", group_keys = False).
            apply(self._cum_single_values))
        
        df_port = (df_purchase_book[
            ["Date", "port_value"]].
            drop_duplicates().
            assign(
                port_change = lambda x: x.port_value.pct_change().fillna(0),
                port_pnl = lambda x: x.port_value * x.port_change,
                cum_pnl = lambda x: np.cumsum(x.port_pnl),
                cum_rtn = lambda x: (np.cumprod(1 + x.port_change) - 1) * 100))
        
        return df_purchase_book, df_port

    def get_weight_and_shares(
        self,
        lookback_window: int,
        rebalance_method: str,
        initial_capital: float):

        df_purchase, df_port = self.get_backtest(
            lookback_window = lookback_window, 
            rebalance_method = rebalance_method,
            initial_capital = initial_capital)

        df_tmp = (df_purchase[
            ["Date", "ticker", "position_value", "port_value", "num_shares"]].
            assign(weight = lambda x: x.position_value / x.port_value).
            drop(columns = ["port_value", "position_value"]).
            pivot(index = "Date", columns = "ticker", values = ["weight", "num_shares"]))

        df_weight, df_shares = df_tmp["weight"], df_tmp["num_shares"]

        return df_weight, df_shares

    def plot_weight_and_shares(
        self,
        lookback_window: int,
        rebalance_method: str,
        initial_capital: float) -> plt.figure:

        df_weight, df_shares = self.get_weight_and_shares(
            lookback_window = lookback_window,
            rebalance_method = rebalance_method,
            initial_capital = initial_capital)

        fig, axes = plt.subplots(ncols = 2, figsize = (20,6))
        (df_weight.assign(
            long_weight = 1).
            drop(columns = [self.long_name]).
            rename(columns = {"long_weight": self.long_name}).
            plot(
                ax = axes[0],
                legend = False, 
                color = "black",
                title = "Weighting",
                ylabel = "%"))

        axes[0].fill_between(
            x = df_weight.index,
            y1 = df_weight[self.short_name],
            y2 = 0,
            where = df_weight[self.short_name] > 0,
            facecolor = "red",
            alpha = 0.3)

        axes[0].fill_between(
            x = df_weight.index,
            y1 = 1,
            y2 = df_weight[self.short_name],
            where = df_weight[self.short_name] < 1,
            facecolor = "green",
            alpha = 0.3)

        (df_shares[
            [self.long_name]].
            plot(
                ax = axes[1],
                color = "green",
                legend = False,
                ylabel = "Num Shares (long)",
                title = "Num of Shares"))

        legend_elements = [
            Patch(facecolor = "green", alpha = 0.3, label = self.long_name),
            Patch(facecolor = "red", alpha = 0.3, label = self.short_name)]

        axes[0].legend(handles = legend_elements)

        axes1_copy = axes[1].twinx()
        axes1_copy.invert_yaxis()
        (df_shares[
            [self.short_name]].
            plot(
                legend = False,
                ax = axes1_copy,
                color = "red"))
        
        axes[1].set_label(self.long_name)
        axes[1].legend(loc = "upper left")
        
        axes1_copy.set_label(self.short_name)
        axes1_copy.legend(loc = "upper right")

        axes1_copy.set_ylabel("Num Shares (short)", rotation = 270, labelpad = 15.5)

        plt.tight_layout()
        return fig
    
    def get_backtest_portfolio_beta_and_notional(
        self,
        lookback_window: int,
        rebalance_method: str,
        initial_capital: float):

        df_purchase, df_port = self.get_backtest(
            lookback_window = lookback_window, 
            rebalance_method = rebalance_method,
            initial_capital = initial_capital)
        
        df_beta = (df_purchase[
            ["Date", "ticker", "directional_beta", "position_value", "port_value"]].
            assign(
                weight = lambda x: x.position_value / x.port_value,
                weighted_beta = lambda x: x.weight * x.directional_beta).
            drop(columns = ["position_value", "port_value", "weight", "directional_beta"]).
            pivot(index = "Date", columns = "ticker", values = "weighted_beta"))
        
        df_port_value = (df_purchase[
            ["Date", "ticker", "position_value"]].
            query("ticker == @self.long_name").
            pivot(index = "Date", columns = "ticker", values = "position_value").
            reset_index().
            merge(
                (df_purchase[
                    ["Date", "port_value"]].
                    drop_duplicates()),
                how = "inner",
                on = ["Date"]).
            set_index("Date"))
        
        return df_beta, df_port_value
    
    def plot_backtest_portfolio_beta_and_notional(
        self, 
        lookback_window: int,
        rebalance_method: str,
        initial_capital: float) -> plt.figure:

        df_beta, df_port_value = self.get_backtest_portfolio_beta_and_notional(
            lookback_window = lookback_window,
            rebalance_method = rebalance_method,
            initial_capital = initial_capital)
        
        fig, axes = plt.subplots(ncols = 2, figsize = (20,6))

        (df_beta[
            [self.long_name]].
            plot(
                ax = axes[0],
                ylabel = "Long Beta",
                legend = False,
                color = ["blue"],
                title = "Portfolio Beta Matching"))

        axes[0].set_label(self.long_name)
        axes[0].legend(loc = "upper right")

        axes0_copy = axes[0].twinx()
        axes0_copy.invert_yaxis()
        (df_beta[
            [self.short_name]].
            plot(
                ax = axes0_copy,
                ylabel = "Short Beta",
                color = ["black"]))

        axes0_copy.set_ylabel("Short Beta", rotation = 270, labelpad = 15.5)
        axes0_copy.legend(loc = "upper left")

        df_port_value.plot(
            ax = axes[1],
            legend = False,
            ylabel = "Notional Value",
            color = "black",
            title = "Notional Value")

        axes[1].fill_between(
            x = df_port_value.index,
            y1 = df_port_value[self.long_name],
            y2 = 0,
            where = df_port_value[self.long_name] > 0,
            facecolor = "green",
            alpha = 0.3)

        axes[1].fill_between(
            x = df_port_value.index,
            y1 = df_port_value["port_value"],
            y2 = df_port_value[self.long_name],
            where = df_port_value[self.long_name] < df_port_value["port_value"],
            facecolor = "red",
            alpha = 0.3)

        legend_elements = [
            Patch(facecolor = "green", alpha = 0.3, label = self.long_name),
            Patch(facecolor = "red", alpha = 0.3, label = self.short_name)]
        axes[1].legend(handles = legend_elements)

        plt.tight_layout()
        return fig
    
    def get_backtest_pnl_and_rtn(
        self,
        lookback_window: int,
        rebalance_method: str,
        initial_capital: float):


        df_purchase, df_port = self.get_backtest(
            lookback_window = lookback_window, 
            rebalance_method = rebalance_method,
            initial_capital = initial_capital)

        df_tmp = (df_purchase[[
            "ticker", "Date", "cum_pnl", "cum_rtn"]].
            pivot(index = "Date", columns = "ticker", values = ["cum_pnl", "cum_rtn"]))

        cum_pnl, cum_rtn = df_tmp["cum_pnl"], df_tmp["cum_rtn"]

        return cum_pnl, cum_rtn
    
    def plot_backtest_pnl_and_rtn(
        self,
        lookback_window: int,
        rebalance_method: str,
        initial_capital: float):

        cum_pnl, cum_rtn = self.get_backtest_pnl_and_rtn(
            lookback_window = lookback_window,
            rebalance_method = rebalance_method,
            initial_capital = initial_capital)
        
        fig, axes = plt.subplots(ncols = 2, figsize = (20,6))

        (cum_pnl[
            [self.long_name]].
            plot(
                ax = axes[0],
                ylabel = "PnL ($)",
                color = "blue"))
        
        ((cum_pnl[[self.short_name]] * -1).
        plot(
            ax = axes[0],
            color = "black"))
        
        (cum_rtn[
            [self.long_name]].
            plot(
                ax = axes[1],
                ylabel = "PnL ($)",
                color = "blue"))
        
        ((cum_rtn[[self.short_name]]).
        plot(
            ax = axes[1],
            color = "black"))

        plt.tight_layout()
        return fig