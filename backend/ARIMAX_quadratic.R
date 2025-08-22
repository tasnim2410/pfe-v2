library(forecast)
library(ggplot2)
library(dplyr)

# --- 1. Data prep: compute lag and nonlinear terms ---
df <- df %>% arrange(year)
df <- df %>%
  mutate(
    pub_lag1 = lag(pub_count, 1)
  ) %>%
  filter(!is.na(pub_lag1)) %>%
  mutate(
    pub_lag1_sq = pub_lag1^2,
    pub_lag1_cu = pub_lag1^3
  )

# --- 2. Train/test split ---
split_year <- 2015
train_idx <- df$year <= split_year
test_idx  <- df$year >  split_year

y_train <- df$patent_count[train_idx]
y_test  <- df$patent_count[test_idx]

# --- Prepare xreg matrices for each model ---
xreg_train_lin  <- as.matrix(df$pub_lag1[train_idx])
xreg_test_lin   <- as.matrix(df$pub_lag1[test_idx])
xreg_train_quad <- as.matrix(df[train_idx, c("pub_lag1", "pub_lag1_sq")])
xreg_test_quad  <- as.matrix(df[test_idx,  c("pub_lag1", "pub_lag1_sq")])
xreg_train_cub  <- as.matrix(df[train_idx, c("pub_lag1", "pub_lag1_sq", "pub_lag1_cu")])
xreg_test_cub   <- as.matrix(df[test_idx,  c("pub_lag1", "pub_lag1_sq", "pub_lag1_cu")])

# --- 3. Fit ARIMAX models on training data ---
mod_lin  <- forecast::auto.arima(y_train, xreg = xreg_train_lin,  stepwise=FALSE, approximation=FALSE)
mod_quad <- forecast::auto.arima(y_train, xreg = xreg_train_quad, stepwise=FALSE, approximation=FALSE)
mod_cub  <- forecast::auto.arima(y_train, xreg = xreg_train_cub,  stepwise=FALSE, approximation=FALSE)

# --- 4. Forecast for the test period ---
h <- length(y_test)
fc_lin  <- forecast::forecast(mod_lin,  xreg = xreg_test_lin,  h = h)
fc_quad <- forecast::forecast(mod_quad, xreg = xreg_test_quad, h = h)
fc_cub  <- forecast::forecast(mod_cub,  xreg = xreg_test_cub,  h = h)

# --- 5. Function to plot residuals (time + ACF) then main plot ---
plot_arimax_result <- function(model, fc, df, train_idx, test_idx, title) {
  years   <- df$year
  y_all   <- df$patent_count
  
  # Residuals: plot first
  print(
    autoplot(ts(model$residuals, start = min(df$year[train_idx]), frequency = 1)) +
      labs(title = paste(title, "- Training Residuals"), y = "Residuals")
  )
  acf(model$residuals, main = paste(title, "- Training Residuals ACF"))
  cat("\nLjung-Box test (training residuals):\n")
  print(Box.test(model$residuals, lag = 10, type = "Ljung-Box"))
  cat("\n")
  
  # Fitted (in-sample) and forecast (out-of-sample)
  fitted_all   <- rep(NA, length(y_all))
  forecast_all <- rep(NA, length(y_all))
  lo95_all     <- rep(NA, length(y_all))
  hi95_all     <- rep(NA, length(y_all))
  fitted_all[train_idx]      <- as.numeric(fitted(model))
  forecast_all[test_idx]     <- as.numeric(fc$mean)
  lo95_all[test_idx]         <- as.numeric(fc$lower[,2])
  hi95_all[test_idx]         <- as.numeric(fc$upper[,2])
  
  plot_df <- data.frame(
    Year = years,
    Actual = y_all,
    Fitted = fitted_all,
    Forecast = forecast_all,
    Lo95 = lo95_all,
    Hi95 = hi95_all
  )
  # Main plot
  p <- ggplot(plot_df, aes(x = Year)) +
    geom_line(aes(y = Actual), color = "black", size = 1.2) +
    geom_line(aes(y = Fitted), color = "purple", size = 1.1, na.rm = TRUE) +
    geom_line(aes(y = Forecast), color = "blue", size = 1.2, na.rm = TRUE) +
    geom_ribbon(aes(ymin = Lo95, ymax = Hi95), fill = "blue", alpha = 0.2, na.rm = TRUE) +
    scale_x_continuous(
      breaks = seq(min(plot_df$Year), max(plot_df$Year), by = 5)
    ) +
    labs(title = title, y = "Patent Count", x = "Year") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, size = 8))
  print(p)
}

# --- 6. Plot and diagnose all models (residuals first, then forecast) ---
plot_arimax_result(mod_lin,  fc_lin,  df, train_idx, test_idx, "ARIMAX Linear (with AR terms)")
plot_arimax_result(mod_quad, fc_quad, df, train_idx, test_idx, "ARIMAX Quadratic (with AR terms)")
plot_arimax_result(mod_cub,  fc_cub,  df, train_idx, test_idx, "ARIMAX Cubic (with AR terms)")

# --- 7. Print accuracy for test set ---
cat("\nTest set accuracy (Linear):\n"); print(accuracy(fc_lin,  y_test))
cat("\nTest set accuracy (Quadratic):\n"); print(accuracy(fc_quad, y_test))
cat("\nTest set accuracy (Cubic):\n"); print(accuracy(fc_cub,  y_test))

# --- 8. Print model summaries for AR/MA term inspection ---
cat("\nSummary (Linear):\n"); print(summary(mod_lin))
cat("\nSummary (Quadratic):\n"); print(summary(mod_quad))
cat("\nSummary (Cubic):\n"); print(summary(mod_cub))

