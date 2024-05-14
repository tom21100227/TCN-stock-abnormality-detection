library(dplyr)
library(ggplot2)
library(tidyverse)

df <- read.csv("data/AAPL_1min_2024-02.csv")

df$Date |> as.POSIXct(format = "%Y-%m-%d %H:%M:%S") -> df$Date

# filter out to just trading hours (9:30am - 4:00pm)
df |> filter(as.numeric(format(Date, "%H")) >= 9 & as.numeric(format(Date, "%H")) < 16) -> df

# check if we have the same amount of observations each day: 
df |> group_by(format(Date, "%Y-%m-%d")) |> summarize(n = n()) # |> filter(n != 390)

# check if the interval between observations is consistent
df |> arrange(Date) |> mutate(diff = c(0, diff(Date))) |> group_by(diff) |> summarize(n = n())

# Sort the df in time order based on date
df |> arrange(Date) -> df

# plot without date, just use index as x-axis

p.no_realtime <- ggplot(df, aes(x = 1:nrow(df), y = Close)) + geom_line() +
  labs(title = "AAPL", x = "Time", y = "Close Price") +
  theme_minimal() + geom_smooth(method = 'gam', formula = y ~ s(x, bs = "cs"))

# facet wrap with each day
p <- ggplot(df, aes(x = Date, y = Close)) + geom_line() +
  labs(title = "AAPL", x = "Time", y = "Close Price") +
  theme_minimal() + # geom_smooth(method = 'gam', formula = y ~ s(x, bs = "cs"), alpha = 0.1) +
  facet_wrap(~format(Date, "%Y-%m-%d"), scales = "free_x") 
p

scale_value <- function(x, min_orig, max_orig, min_new, max_new) {
  y <- min_new + ((x - min_orig) * (max_new - min_new) / (max_orig - min_orig))
  return(y)
}

# scale volume to be within the range of close price for plotting purpuse
df |> mutate(Volume.adj = scale_value(Volume, min(Volume), max(Volume), min(Close), max(Close))) -> df

# facet wrap each day, add volume of each data point to the bottom, use separate axis for line and bar
p_volume <- ggplot(df, aes(x = Date)) + geom_line(aes(y = Close), color = 'black') +
  geom_line(aes(y = Volume.adj), stat = "identity", color = "darkgreen", alpha = 0.5) +
  labs(title = "AAPL", x = "Time", y = "Close Price") +
  theme_minimal() + geom_smooth(aes(y = Close), method = 'loess', color = "orange", span = 0.1, linewidth = 0.5) +
  facet_wrap(~format(Date, "%Y-%m-%d"), scales = "free_x") 
p_volume

