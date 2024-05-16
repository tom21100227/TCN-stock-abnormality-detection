library(dplyr)
library(ggplot2)

# Load the data
full_data <- read.csv("data/5-min_data.txt")



# Filter the data
tickers = c("^DJI", "^SPX")
# full_data$X.DATE. <- as.Date(full_data$X.DATE., format = "%Y%m%d")
full_data |> filter(X.TICKER. %in% tickers) |> filter(X.DATE. == 20240325) -> sub.data

p <- ggplot(sub.data, aes(x = X.TIME., y = X.CLOSE.)) + geom_line() +
  labs(title = "Stock Market Indexes", x = "Date", y = "Close Price") +
  theme_minimal() + facet_wrap(~X.TICKER., scales = "free_y")
p
