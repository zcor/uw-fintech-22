# Module 11: Virtual Class I Lesson Plan (2 hours)

## Overview

The goal of this session is to introduce students to time series visualization and time series data management techniques.

Prior to the two hour class time starting, there will be 30 minutes of office hours.

In the first section of class, the students will be introduced to forms of financial time series data, and how to visualize and summarize them.

In the last section of class, the students will be introduced to the concept of model building with time series data.

## Learning Objectives

At the end of the session, learners will be able to:

* Understand what time-series data is, and where it occurs in finance
* Group and analyze time-series data for useful patterns
* Visualize time-series data using HvPlot

---

## Time Tracker

| Start   | #  | Activity                                               | Time |
| ------- | -- | ------------------------------------------------------ | ---- |
| 7:00 PM | 1  | Instructor Do: Welcome                                 | 0:05 |
| 7:05 PM | 2  | Instructor Do: Time-Series Warm-up                     | 0:10 |
| 7:15 PM | 3  | Instructor Do: Grouping with Time Data                 | 0:10 |
| 7:25 PM | 4  | Student Do: Grouping by Datetime                       | 0:15 |
| 7:40 PM | 5  | Instructor Review: Grouping by Datetime                | 0:05 |
| 7:45 PM | 6  | Instructor Do: Visualizing Time-Series with hvPlot     | 0:10 |
| 7:55 PM | 7  | Student Do: Visualizing Time-Series with hvPlot        | 0:15 |
| 8:10 PM | 8  | Instructor Review: Visualizing Time-Series with hvPlot | 0:10 |
| 8:20 PM | 9  | Instructor Do: Analyzing Time Series Data              | 0:15 |
| 8:35 PM | 10 | Student Do: Analyzing Time Series Data                 | 0:20 |
| 8:55 PM | 11 | Instructor Review: Analyzing Time Series Data          | 0:05 |
| 9:00 PM | 12 | END                                                    |      |

---

## Instructor Do: Office Hours (0:30)

Welcome to Office Hours! Remind the students that this is their time to ask questions and get assistance from their instructional staff as they’re learning new concepts and working on the challenge assignment. Feel free to use breakout rooms to create spaces focused on specific topics, or to have 1:1s with students.

Expect that students may ask for assistance such as the following:

* Challenge assignment.
* Further review on a particular subject.
* Update issues with PyViz or Jupyter extensions.
* Debugging assistance - rendering images with PyViz can sometimes be a challenge!
* Help with computer issues.
* Guidance with a particular tool.

---

## Class Activities

### 1. Instructor Do: Welcome and Temperature Check (5 min)

* Welcome students back to for week 11 of class.

* Gauge student sentiment about Module 10, especially how they feel their understanding of what machine learning models are and what they are used for.

* Let them know that Module 11 is going continue this approach of using models to predict outcomes: The difference with these time series models is that we're usually focusing on predicting just one thing-- the next day's price of a single stock, for example.

Explain that, before we can get to the specialized models for time-series prediction, we must first get comfortable with using datetime functionality to visualize those time-series datasets.

---
### 2. Everyone Do: Time-Series Warm-Up (10 min)

* Since this session is all about time-series, let's begin with a warm up of some Pandas time-series functions.

* Ask students to open the starter code, and ask them to code along while you live code these time-series examples.

**Files:**

[Resources](Activities/01-Ins_Time_Series_Warmup/Resources)

[Starter Code](Activities/01-Ins_Time_Series_Warmup/Unsolved/time-series-warmup.ipynb)

* Much of time-series work in python and pandas relies on declaring a `datetime` index. When we read in the `amazon.csv` data, we'll declare the `datetime` index up front, telling pandas to automatically recognize these dates.

```python
# Import data
amzn_path = Path('../Resources/amazon.csv')

# Read in data and index by date
df = pd.read_csv(
    amzn_path,
    index_col='Date',
    parse_dates=True,
    infer_datetime_format=True
)
```

* Once we have a `datetime` index, we can do useful things, like create subsets of the dataframe based on specific dates. For example, we can create a dataframe that includes data from just September, 2018:

```python
# Select all rows from September 2018
sep_2018 = df.loc['2018-09']
sep_2018.tail()
```

* Plotting is also easy with a `datetime` index. Inbuilt pandas plotting functions can work directly on dataframes, such as a line plot:

```python
# Plot the closing prices using a line plot
df.Close.plot()
```

* Besides slicing and plotting, another important skill is to be able to `resample`  datetime indexed data. Resampling means that we can aggregate data, to get a broader viewpoint on it. For example, we can convert our daily Amazon price data over to weekly average prices:

```python
# Resample the closing prices to weekly and take the mean
weekly = df['Close'].resample('W').mean()
weekly.head()
```

Plotting the result, we see that looking at the weekly average price is a lot smoother than it's daily dataframe counterpart. This is one of the benefits of resampling: sometimes it's easier to quickly identify bigger picture time-series patterns.

```python
# Plot the weekly average closing prices as a line chart
weekly.plot()
```

Ask students if there are any questions before moving on.

### 3. Instructor Do: Grouping with Time Data (10 min)

In this activity you will walk students through the steps of grouping-time series data in order to quickly summarize and understand it.

> "Grouping times-series data into various aggregations is important as it allows the analyst to visually hunt for patterns that they can then insert into a model. It's also helpful when it comes to explaining time-series results to others, as people not familiar with interpreting output from models will still be familiar with interpreting visualizations of repeatable patterns."

**Files:**

[Resources](Activities/02-Ins_Time_Groupby/Resources)

* **Groupby** is the process of grouping data in order to better summarize it.

* We touched on Groupby for data visualization in Module 6. This time, we'll group by time in a number of ways.

* We can groupby time to aggregate -- if we see that a particular moment in time is above or below this aggregation, that tells us something about how unusual that point in time is.

* The `groupby` function accepts a **list** of dimension(s) to group by as an argument.

* After a `groupby` function, we attach an `aggregation method` to aggregate across this dimension(s)

* For example, we might want to group daily stock returns by each week in the dataset, calculating the weekly average.
  * The **list** of dimensions will be each week in the time series.
  * The `aggregation method` will be the `mean()` function.

Open the starter file and live code the following examples:

* We've seen before that if we have a datetime index, we can slice it to specific sub-dates to make new dataframes.

```python
# Slice the Data to One Specific Month
volume_jan_2021 = tsla_data.loc['2021-01-01':'2021-01-31']
volume_jan_2021
```

  * Highlight how, based on this sub-date DataFrame, we can perform aggregation functions. Here, we sum the total number of shares traded for the month of January, 2021:

```python
# Calculate the total number of shares traded for the month of January, 2021
jan_2021_volume = volume_jan_2021['volume'].sum()
jan_2021_volume
```

* Rather than manually create a subset DataFrame for each month though, we can perform these aggregation functions all at once using Groupby.

* For example, we take advantage of the fact that we have a datetime index, and use the `year` and the `month` of that index as our GroupBy arguments.

```python
# Specify the way you want to group things -- here we are using the datetimeindex
groupby_levels = [tesla_volume.index.year,tesla_volume.index.month]

# Then Groupby that, choosing an aggregation function
total_monthly_volume = tesla_volume.groupby(by= groupby_levels ).sum()
total_monthly_volume
```

  * Just to illustrate how this might be used to identify time-series patterns, let's compare our initial January 2021 volume to the median monthly volume. The median monthly volume can be calculated by further aggregating the total monthly volume DataFrame we calculated via GroupBy:

```python
# We can do summary statistics on the aggregated data we just created
# For Example: View the median amount of monthly shares traded
median_monthly_volume = total_monthly_volume.median()
median_monthly_volume

# Compare the shares traded in January 2021 to the median amount that get traded each month
jan_2021_volume/median_monthly_volume
```

* What we see from this is that January 2021 volume was unusually low for a typical month of trading in Tesla: volume for that month was about 88% of the median volume of shares traded over the full 2010-2020 period.

Ask students if they have any questions about the process of grouping by a datetime index before moving on to the activity.

### 4. Student Do: Grouping by Datetime (10 min)

In this short activity, students will get the opportunity to practice grouping by DateTime indices and analyzing the results.

Slack out the following files to the students.

**Files:**

[Instructions](Activities/03-Stu_Groupby/README.md)

[Starter Code](Activities/03-Stu_Groupby/Unsolved/spy-groupby.ipynb)

**Instructions:**

In this exercise you will practice grouping financial data by its DateTime index and analyzing the results.

Using the [starter code](Activities/03-Stu_Groupby/Unsolved/spy-groupby.ipynb) provided, perform the following:

1. Import the data, taking care to declare the `datetime` index.

2. Slice the data to one specific month--say, last December, March, or January.

3. Save the total volume of shares traded for that month into a new variable.

4. Group the bigger dataset on share volume into `year` and `month` using the `datetime` index. Use this grouping to create a DataFrame of total monthly SPY shares traded each month.

5. Using the DataFrame constructed in step (4), Calculate the `median` monthly total volume of shares traded in the S&P 500.

6. Compare this `median` number to the number you calculated in step (3). How does that month compare in terms of trading activity?

### 5. Instructor Review: Grouping by Datetime (5 min)

**Files:**

[Solution code](Activities/03-Stu_Groupby/Solved/spy-groupby.ipynb)

Review the process of grouping `datetime` index data with the students:

* Note that the process for summarizing and grouping datetime is very general. The simplest approach is to slice to a period we're interested in and summarize (**Note:** based on the instructions, students were free to choose a different month to slice):

```python
# Slice the Data to One Specific Month
volume_jan_2021 = spy_data.loc['2021-01-01':'2021-01-31']
volume_jan_2021
```

* Aggregating in this case amounts to calling one additional function:

```python
# Calculate the total number of shares traded for the month of January, 2021
jan_2021_volume = volume_jan_2021['volume'].sum()
jan_2021_volume
```

Usually though, we want to do this aggregation across groups. So with the code below, we are effectively slicing and aggregating across multiple time groups at once:

```python
# Specify the way you want to group things -- here we are using the datetimeindex
groupby_levels = [spy_volume.index.year,spy_volume.index.month]

# Then Groupby that, choosing an aggregation function
total_monthly_volume = spy_volume.groupby(by= groupby_levels ).sum()
total_monthly_volume
```

From here, we can compare and contrast what a typical month looks like for the S&P (lots of shares traded!) versus the month that we sliced:

```python
# We can do summary statistics on the aggregated data we just created
# For Example: View the median amount of monthly shares traded
median_monthly_volume = total_monthly_volume.median()
median_monthly_volume

# Compare the shares traded in January 2021 to the median amount that get traded each month
jan_2021_volume/median_monthly_volume
```

> "Pandas grouping, aggregating and filtering all come into play when we need to analyze time data. Now that we can apply these skills to time-series data, let's see if we can combine that with some visualization techniques to strengthen our analysis capability."

### 6. Instructor Do: Visualizing Time Series with hvPlot (10 min)

> "Now that we can split and apply summary statistics to time-series data, let's see if we can plot those results visually."

> "In FinTech it is quite common to present analysis not as a stream of numbers or tables, but rather in visual form. Understanding how to create and interpret visual patterns in time-series data is thus extremely important."

Use the following files to live code the provided examples for visualizing grouped data.

**Files:**

[Starter code](Activities/04-Ins_Visualizing_Time_Patterns/Unsolved/visualizing_time_patterns.ipynb)

[Solution code](Activities/04-Ins_Visualizing_Time_Patterns/Solved/visualizing_time_patterns.ipynb)

* Before jumping straight into building models to make time-series predictions, it's common to visualize the data.

* If you visualize first, you might be able to spot clear patterns, or get ideas for new features to add to a time-series model. You might also be able to spot any errors in the data early on.

* We've done visualization before using `hvPlot`; this time we'll combine `hvPlot` with time-series groupings.

Open the [starter code](Activities/04-Ins_Visualizing_Time_Patterns/Unsolved/visualizing_time_patterns.ipynb) and review the initial imports and reading in of the DataFrame with the students.

Start to live code using the following examples to visualize time-series data with hvPlot.

* Time-series data usually focuses on the measurement of just a single column over time. For the plots that follow, it'll be best to slice to just the column we are interested in: Tesla hourly share volume.

```python
# It's usually easier if we plot a single time-series
tsla_volume = tsla_data['volume']
```

* When we do a `GroupBy` plot in hvPlot, the first thing we need to do is determine what to actually group by. Let's try using the `DateTimeIndex` function `dayofweek`, to analyze the results by the day of the week (Monday, Tuesday, Wednesday, etc...):

```python
# Declare the group level to be the day of the week (e.g., Mon, Tues,...)
group_level = tsla_data.index.dayofweek
```

* Once we've declared the way that we want to group the data, what's left is to declare how to aggregate, and what type of plot to call. Here, we'll calculate the average hourly share volume by the day of the week, and plot it using a default hvplot line plot:

```python
# Plot average daily volume according to day of the week
tsla_volume.groupby(group_level).mean().hvplot()
```

* Notice how there's a lot more share trading in Tesla on a Monday, compared to the rest of the week?

   > Analysis Tip: If we were building a time series model to predict trading activity, one variable we might want to add into the model is therefore whether or not it's Monday.

* One of the cool things about hvPlot and grouping time-series data is that you can use it to quickly identify more complex patterns.

* Take for example the code below. This uses the `DateTimeIndex` to drill down deeper into the day-of-week pattern we just looked at. Within each day of the week and hour, we can identify hot spots of intense share trading activity:

```python
# Use hvPlot to visualize the hourly trends across days of week in a heatmap
tsla_volume.hvplot.heatmap(x='index.hour', y='index.dayofweek', C='volume', cmap='reds').aggregate(function=np.mean)
```

* Notice how we have to include the `.aggregate(function=np.mean)` code at the end. Here we're plotting the average by hourly-by-weekday price (across all days in the dataset), but we could just as easily look at the median or max, too.

* Notice how (except for 10 AM), volume on Mondays tends to be concentrated in the late afternoon, but that the reverse is true on Tuesdays?

* Let's try one more plot--this time more macro in scale. Specifically, let's visualize whether there's a particular time of the year in which Tesla shares tend to be the most heavily traded.

* To do this, we call the default line plot in `hvplot()`, but specify our grouping as `weekofyear`. Aggregating by `mean`, we'll be able to identify any patterns in average hourly share volume, according to what week of the year it is (e.g., 1st week of the year, versus 52nd week of the year):

```python
# Group the hourly search data to plot (use hvPlot) average volume by the week of the year
tsla_volume.groupby(tsla_volume.index.weekofyear).mean().hvplot()
```

Ask students if they have any questions about the `hvplot` functions and on grouping or aggregating by `datetime` objects.

### 7. Student Do: Visualizing Time-Series with hvPlot (15 min)

In this activity, students will work to visualize and analyze time-series patterns in the S&P 500 volume data.

Slack out the starter code and instructions to the students.

**Files:**

[Instructions](Activities/05-Stu_Visualizing_Time_Patterns/README.md)

[Starter code](Activities/05-Stu_Visualizing_Time_Patterns/Unsolved/visualizing_time_patterns.ipynb)

**Instructions:**

In this activity you will gain some additional practice working with the hvPlot grouped plots.

Using the [starter code](Activities/05-Stu_Visualizing_Time_Patterns/Unsolved/visualizing_time_patterns.ipynb) provided, complete the following:

1. Read the S&P 500 volume into a DataFrame. (Make sure to declare the datetime index).

2. Slice the dataframe so that it just includes the volume data.

3. Using hvPlot, plot the volume data according to the day of the week to answer the following question: In what day does trading in the S&P500 tend to be the most active?

4. Next, use hvPlot to visualize hourly trends for each day of the week in the form of a heatmap. Based on this, does any day-of-week pattern concentrate in just a few hours of a particular day?

5. Lastly, create a plot with hvPlot that shows the data grouped by the calendar week in the year (week of year). Does share trading intensity tend to increase at any particular time of the calendar year?

### 8. Instructor Do: Review Visualizing Time Series with hvPlot (10 min)

Review the visualization activity, focusing on the the various ways that grouped time data plots can tell a story, or set the stage for more informed analysis.

In the activity, the `S&P 500` dataset was the chosen data subset. (As a reminder, this is an ETF on the largest 500 public firms in the U.S., so it is a barometer for the condition and activity in the broader economy).

* Import Libraries and dependencies

```python
import pandas as pd
from pathlib import Path
import hvplot.pandas
%matplotlib inline
```

* Import data and read the CSV file into a Dataframe accounting for date/time considerations

```python
# Import data
spy_path = Path('../Resources/spy_stock_volume.csv')

# Read the S&P 500 volume into a DataFrame. (Make sure to declare the datetime index).
spy_data = pd.read_csv(spy_path, index_col='Date',
            parse_dates=True, infer_datetime_format=True)
spy_data
```

* Slice the dataframe so that it just includes the volume data.

```python
spy_volume = spy_data['volume']
```

* Declare the group level to be the day of the week (e.g., Mon, Tues,...), then plot the average daily volume according to the day of the week.

```python
# Declare the group level to be the day of the week (e.g., Mon, Tues,...)
group_level = spy_volume.index.dayofweek
# Plot average daily volume according to day of the week
spy_volume.groupby(group_level).mean().hvplot()
```

* Based on the above, it looks like later in the week (Wednesday through Friday) tends to be the time when share trading activity is at its highest.

* Next, Use hvPlot to visualize the hourly trends across days of week in a heatmap.

```python
spy_volume.hvplot.heatmap(x='index.hour', y='index.dayofweek', C='volume', cmap='reds').aggregate(function=np.mean)
```

* By looking at where the darker region on the heatmap is placed, it's clear that when it comes to S&P 500 trading, most share trading really seems to be concentrated in the last hour or two of the day. Other than possibly Mondays (the "hottest" point in the heatmap), this effect doesn't seem to depend much on the day of the week.

* Finally, group the data by the calendar week in the year (week of year).

```python
spy_volume.groupby(spy_volume.index.weekofyear).mean().hvplot()
```

* Based on the resulting graph from the code above, it looks like share activity generally stays constant throughout the year, except for somewhere around the 11th-14th week (roughly March/April). This is about the time that many company Annual Reports (presentation and discussion of annual financial results) come out, so this could be one reason why we are seeing a consistent volume pattern around this time.

Ask students if there are any questions about hvPlot or the process of analyzing the visual output of time-series data.

  ---

### 9. Instructor Do: Analyzing Time Series Data (Correlation) (15 min)

Let the students know that the next topic we are going to tackle is `time-series correlation`.

> Looking at time-series correlation amounts to looking at predictive relationships. For example, if one variable moves up over the next hour, what does that spell for another variable an hour later?

> In time-series analysis, sometimes we even use past values of a variable to predict its future value. For example, if the stock return was high over the last hour, does that mean it's also likely to be high one hour later?

> As we've seen in a previous module, one way to quickly identify variable relationships is through **correlation**. The key difference here, however, is in order to identify any predictive relationship, we want to know the correlation of variables measured at different points in time.

**Files:**

[Starter Code](Activities/06-Ins_Predicting_With_Correlation/Unsolved/predicting_with_correlation.ipynb)

[Solution Code](Activities/06-Ins_Predicting_With_Correlation/Solved/predicting_with_correlation.ipynb)

Using the [starter file](Activities/06-Ins_Predicting_With_Correlation/Unsolved/predicting_with_correlation.ipynb), live code as much of the explanation as possible.

The file contains the same Tesla volume and price data. So far, we've just used the volume column. Now, we'll use both volume and closing prices to try and identify some time-series predictive correlations.

* Start by just plotting the remarkable closing price of Tesla over time, using hvPlot. Wouldn't it have been nice if we could have predicted some of this action!

```python
# Use hvPlot to visualize the closing price of Tesla over time.
tsla_data['close'].hvplot()
```

  * Let's narrow in on a specific window, say, the full year of 2020. This was a special time for Tesla's stock, not too mention the markets overall.

```python
tesla_2020 = tsla_data['2020-01':'2020-12']
```

* Let's see if we can use this slice to first visually identify whether there's any relationship between volume and price. This time, we'll call upon a new visualization method, but still using hvPlot:

```python
tesla_2020.hvplot(shared_axes=False, subplots=True).cols(1)
```

* The code above takes the columns in the data, and plots them as separate graphs. Since all the columns have the same datetime index though, each graph is automatically aligned with the other.

* Stacking plots like this (`subplots=True`) is one way to quickly identify potential variable relationships, without having to deal with visual distortions that can occur when variables are on a different scale.

* Based on the graph, it looks like there may be increased trading activity surrounding increases in the share prices, although it's not clear yet whether this is a predictive relationship (i.e., whether current volume predicts next period's price).

* To better identify whether or not there's a predictive, or "lead-lag" type of relationship, we can code up a version of the `volume` variable which is lagged by one period (one hour):

```python
# Create a new volume column that shifts the volume back by one hour
tsla_data['Lagged Volume'] = tsla_data['volume'].shift(1)
```

> Note: `shift(1)` -- a positive `1`-- means that we're pulling the previous hour's volume data. If we used `shift(-1)`, we'd be using the volume data from one period in the future instead.

* We're going to use this column we just constructed to predict the next hour's return. Before we do that though, let's create two more columns: Stock Volatility and Hours Stocky Return.

* The code below creates an estimate of rolling stock volatility for Tesla. With this, we're looking at the short-term risk of investing the stock.

```python
tsla_data['Stock Volatility'] = tsla_data['close'].pct_change().rolling(window=4).std()
```

* In addition, the hourly percentage stock return for Tesla:

```python
tsla_data['Hourly Stock Return'] = tsla_data['close'].pct_change()
```

* We'll want to see whether lagged trading volume can predict the next hour's stock returns, or its near-term volatility. We can do that by looking at correlation:

  ```python
  tsla_data[['Stock Volatility', 'Lagged Volume', 'Hourly Stock Return']].corr()
  ```

* What we're really interested in is the `Lagged Volume` column: this tells us whether levels of share trading this hour can predict either `Stock Volatility` or the `Hourly Stock Return` over the following hour.

* That column shows that while we can use volume to predict changes in risk (volatility) for the stock it may not be as useful for predicting the next hour's percentage return.

Ask the students if they have any questions about calculating or interpreting time series correlation, or the new hvPlot graph we just learned.

### 10. Student Do: Analyzing Time Series Data (20 min)

In this activity, students will examine the performance of a stock over time using multi-indexing with dates.

Slack out the following files to the students.

**Files:**

[Instructions](Activities/07-Stu_Predicting_With_Correlation/README.md)

[Starter code](Activities/07-Stu_Predicting_With_Correlation/Unsolved/predicting_with_correlation.ipynb)

[Resources](Activities/07-Stu_Predicting_With_Correlation/Resources)

**Instructions:**

In this activity students will get practice with the creating an hvPlot dual axis plot to juxtapose time-series data. They'll then use this and other data to analyze lead-lag relationships using time-series correlation.

Using the [starter code](Activities/07-Stu_Predicting_With_Correlation/Unsolved/predicting_with_correlation.ipynb) provided, complete the following steps.

1. Read in the S&P 500 stock volume and price data.

2. Plot S&P 500's performance over time using an hvPlot line plot.

3. Based on this plot, slice to just a few months to where the market seems to have suffered a big decline. (This is meant to be a little subjective; pick the time you think is most volatile/downward).

*For steps (4)-(10) that follow, just analyze this downward sub-period DataFrame.*

4. Using this downward sub-period, use hvPlot's ability to create two graphs, stacked one on top of each other. Specifically, plot the hourly `close` price and hourly `volume` of shares traded. Looking at this visual, does it apper there is any relationship between `volume` and `close`?

5. Create a column called `Lagged Volume`, which is the `volume` column, but shifted back in time by one hour.

6. Create another column called `Stock Volatility`, which is the rolling standard deviation of SPY's stock price returns. (Consider using a 4-hour moving average, or experiment with your own horizon to see if it impacts predictability).

7. Last (but not least!), construct a column called `Hourly Stock Return`, which is the percentage return on the S&P at each hour.

8. Using these three columns, construct a `correlation` table, and answer the following questions:

    * Does this hours trading volume predict the next hour's market volatility?

    * Does this hours trading volume predict the next hour's market return?

### 11. Instructor Review: Analyzing Time Series Data (5 min)

As you review the code for this activity with the students, be sure to cover the major themes of the day: concatenation, groupby and the related aggregators and the DateTimeIndex attributes.

This activity did not examine the DateTimeIndex  index attribute of `index.day`, but live code in an example if the opportunity presents itself.

Also, reiterate that calculating performance over different time periods is quite common in finance and FinTech. Being able to easily manipulate these time periods as well as the DataFrames will be a value added skill in their arsenal.

**Files:**

[Solution code](Activities/07-Stu_Predicting_With_Correlation/Solved/predicting_with_correlation.ipynb)

[Resources](Activities/07-Stu_Predicting_With_Correlation/Resources)

* Read in the S&P 500 data, declaring the date column as a `datetime` index.

```python
# Import data
spy_path = Path('../Resources/spy_stock_volume.csv')

# Read in data and index by date
spy_data = pd.read_csv(spy_path, index_col='Date', parse_dates=True, infer_datetime_format=True)
```

* Use hvPlot to visualize the closing price of the S&P 500 over time.

```python
spy_data['close'].hvplot()
```

* Slice to a downward or more volatile period (here, we've chosen August through September. Students may have chosen an alternative time period to study).

```python
# Slice to a volatile/downward trend period
downward_period = spy_data['2020-08':'2020-10']

# Preview the result (your results may vary depending on period selected)
downward_period
```

* Use hvPlot to visualize the close and volume data, one graph above the other:

```python
# Use hvPlot to visualize the close and volume data
downward_period.hvplot(shared_axes=False, subplots=True).cols(1)
```

* Create the column we're interested in testing for a lead-lag relationship: `Lagged Volume`:

```python
# Create a new volume column
# This column should shift the volume back by one hour
downward_period['Lagged Volume'] = downward_period['volume'].shift(1)
```

* Create a column called `Stock Volatility`, which is the rolling average standard deviation of the S&P 500 closing price.

* As well, create a third column, which is the `Hourly Stock Return`:

```python
# This column should calculate the standard deviation of the closing stock price return data over a 4 period rolling window
downward_period['Stock Volatility'] = downward_period['close'].pct_change().rolling(window=4).std()

# This column should calculate hourly return percentage of the closing price
downward_period['Hourly Stock Return'] = downward_period['close'].pct_change()
```

* With these three columns, we can construct a correlation table and look for time-series relationships.

```python
# Construct correlation table of Stock Volatility, Lagged Volume, and Hourly Stock Return
downward_period[['Stock Volatility', 'Lagged Volume', 'Hourly Stock Return']].corr()
```

* Based on the time horizon selected above, the correlation table suggests a predictive relationship between `Lagged Volume` and `Stock Volatility`: as volume rises, so does volatility over the next hour. The low correlation between `Lagged Volume` and `Stock Return`, however suggests that simply looking at volume to predict the percentage return one hour later is not going to work.

* This is actually typical: for large publicly traded stocks, simple predictive relationships like this can be hard to exploit profitably, because there are already many professional traders actively trading in the stock. More obscure assets (small cryptocurrencies or penny stocks, for example) are more likely to exhibit predictable behavior.

Confirm that students understand how all of these pieces for working with time-series data fit together to form an cohesive analysis for understanding and predicting data. Ensure them that understanding these pieces will help them as they turn their attention to using cutting edge forecasting models later in the module.

---

## Open Office Hours

### Q&A and System Support

This is an opportunity to support students in any way that they require.

* Ask the students if they have any questions about the material covered in today's live lesson.
* Ask students if they have any questions about the material covered in the async content.
* Review the Challenge Lesson with the students and remind them of the due date.

---

© 2021 Trilogy Education Services, a 2U, Inc. brand. All Rights Reserved.
