#!/usr/bin/env Rscript
# arimax_linear.R
# ARIMAX with linear exog: pub_{t-1}
# Truncation: patents -> drop last 3 years; publications -> drop last 1 year
# Modes:
#   - test   : train on all but last 5 years, test on last 5 years, report MAE/RMSE
#   - future : train all (after truncation), forecast H with future pub path
#
# Output: compact JSON on stdout. Diagnostics go to stderr via message().
# Deps: DBI, RPostgres, dplyr, forecast, jsonlite

suppressPackageStartupMessages({
  library(DBI)
  library(RPostgres)
  library(dplyr)
  library(forecast)
  library(jsonlite)
})

# ---------- args/env helpers ----------
args <- commandArgs(trailingOnly = TRUE)
parse_flag <- function(flag, default = NULL) {
  hit <- grep(paste0("^", flag, "="), args, value = TRUE)
  if (length(hit) == 0) return(default)
  sub(paste0("^", flag, "="), "", hit[1])
}
get_cfg <- function(flag, env, dflt) {
  val <- parse_flag(flag, NA)
  if (!is.na(val) && nzchar(val)) return(val)
  envv <- Sys.getenv(env, "")
  if (nzchar(envv)) return(envv)
  dflt
}

# ---------- config (flags first, then env, then defaults) ----------
db_name <- get_cfg("--dbname", "PG_DB", "patent_db")
db_host <- get_cfg("--host",   "PG_HOST", "localhost")
db_port <- as.integer(get_cfg("--port",   "PG_PORT", "5433"))
db_user <- get_cfg("--user",   "PG_USER", "postgres")
db_pass <- get_cfg("--password","PG_PASSWORD", "")
sslmode <- get_cfg("--sslmode","PG_SSLMODE","prefer")  # prefer|require|disable|allow

pat_table <- get_cfg("--pat_table","PAT_TABLE","raw_patents")
pub_table <- get_cfg("--pub_table","PUB_TABLE","research_data3")

mode        <- tolower(get_cfg("--mode", "MODE", "test"))      # test|future
split_year  <- as.integer(get_cfg("--split_year", "SPLIT_YEAR", "2020")) # ignored in test mode
horizon     <- as.integer(get_cfg("--horizon", "HORIZON", "5"))
pub_future_years  <- get_cfg("--pub_future_years",  "PUB_FUTURE_YEARS", "")
pub_future_values <- get_cfg("--pub_future_values", "PUB_FUTURE_VALUES","")
pub_strategy      <- tolower(get_cfg("--pub_future_strategy","PUB_FUTURE_STRATEGY","linear")) # linear|flat
DEBUG <- as.integer(get_cfg("--debug", "DEBUG", "0"))


# ---------- small utils ----------
safe_int <- function(x) suppressWarnings(as.integer(x))
safe_num <- function(x) suppressWarnings(as.numeric(x))
mae <- function(y, yhat) mean(abs(y - yhat))
rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))
linear_extrapolate <- function(years, vals, h) {
  n <- length(years)
  if (n < 2) return(rep(tail(vals, 1), h))
  slope <- (vals[n] - vals[1]) / max(1, (years[n] - years[1]))
  fut_years <- seq(years[n] + 1, years[n] + h, by = 1)
  fut_vals  <- vals[n] + slope * seq_len(h)
  list(years = fut_years, values = fut_vals)
}

# ---------- debug banner (stderr) ----------
if (DEBUG == 1) {
  pw_set <- ifelse(nzchar(db_pass), "YES", "NO")
  message(sprintf(
    "[ARIMAX-linear] DB: dbname=%s host=%s port=%d user=%s password_set=%s sslmode=%s",
    db_name, db_host, db_port, db_user, pw_set, sslmode
  ))
  message(sprintf("[ARIMAX-linear] Tables: patents=%s, pubs=%s", pat_table, pub_table))
  message(sprintf("[ARIMAX-linear] Mode=%s split_year=%d horizon=%d strategy=%s", mode, split_year, horizon, pub_strategy))
}

# ---------- connect ----------
con <- tryCatch(
  dbConnect(
    RPostgres::Postgres(),
    dbname = db_name, host = db_host, port = db_port,
    user = db_user, password = db_pass,
    sslmode = sslmode
  ),
  error = function(e) {
    msg <- paste("DB connection failed:", conditionMessage(e))
    if (DEBUG == 1) message("[ARIMAX-linear] ", msg)
    cat(toJSON(list(ok = FALSE, stage = "connect", error = msg), auto_unbox = TRUE))
    quit(save = "no", status = 0)
  }
)
on.exit(try(dbDisconnect(con), silent = TRUE))


# --- add this helper near the top (after library calls) ---
parse_database_url <- function(url) {
  m <- regexec("^postgres(?:ql)?(?:\\+[^:]*)?://([^:]+):([^@]+)@([^:/]+):?(\\d+)?/([^/?#]+)", url)
  parts <- regmatches(url, m)[[1]]
  if (length(parts) == 6) {
    list(
      user = parts[2],
      password = parts[3],
      host = parts[4],
      port = ifelse(is.na(parts[5]) || parts[5] == "", "5432", parts[5]),
      dbname = parts[6]
    )
  } else {
    stop("Could not parse DATABASE_URL")
  }
}

# --- drop this right after your existing config vars ---
db_url <- Sys.getenv("DATABASE_URL", "")
if (nzchar(db_url)) {
  p <- parse_database_url(db_url)
  db_user <- p$user
  db_pass <- p$password
  db_host <- p$host
  db_port <- as.integer(p$port)
  db_name <- p$dbname
}



# ---------- queries (parametrized table names) ----------
q_pat <- sprintf("
  SELECT first_filing_year::int AS year, COUNT(*)::int AS patent_count
  FROM %s
  WHERE first_filing_year IS NOT NULL
  GROUP BY first_filing_year
  ORDER BY first_filing_year;", pat_table)

q_pub <- sprintf("
  SELECT year::int AS year, COUNT(*)::int AS pub_count
  FROM %s
  WHERE year IS NOT NULL
  GROUP BY year
  ORDER BY year;", pub_table)

pat_df <- tryCatch(dbGetQuery(con, q_pat),
  error = function(e) {
    msg <- paste("Query patents failed:", conditionMessage(e))
    if (DEBUG == 1) message("[ARIMAX-linear] ", msg)
    cat(toJSON(list(ok = FALSE, stage = "query_patents", error = msg), auto_unbox = TRUE))
    quit(save = "no", status = 0)
})
pub_df <- tryCatch(dbGetQuery(con, q_pub),
  error = function(e) {
    msg <- paste("Query publications failed:", conditionMessage(e))
    if (DEBUG == 1) message("[ARIMAX-linear] ", msg)
    cat(toJSON(list(ok = FALSE, stage = "query_publications", error = msg), auto_unbox = TRUE))
    quit(save = "no", status = 0)
})

if (DEBUG == 1) {
  message(sprintf("[ARIMAX-linear] Rows: patents=%d, pubs=%d", nrow(pat_df), nrow(pub_df)))
}

if (nrow(pat_df) == 0L) {
  cat(toJSON(list(ok = FALSE, stage = "no_patent_rows", error = "No rows from patents table."), auto_unbox = TRUE))
  quit(save = "no", status = 0)
}
if (nrow(pub_df) == 0L) {
  cat(toJSON(list(ok = FALSE, stage = "no_publication_rows", error = "No rows from publications table."), auto_unbox = TRUE))
  quit(save = "no", status = 0)
}

# ---------- asymmetric truncation ----------
pat_cut <- max(pat_df$year, na.rm = TRUE) - 3L   # drop last 3 years from patents
pub_cut <- max(pub_df$year, na.rm = TRUE) - 1L   # drop last 1 year from pubs
pat_df <- dplyr::filter(pat_df, year <= pat_cut)
pub_df <- dplyr::filter(pub_df, year <= pub_cut)

# ---------- align, fill, guard ----------
years_all <- sort(intersect(pat_df$year, pub_df$year))
if (length(years_all) < 8) {
  cat(toJSON(list(ok = FALSE, stage = "align", error = "Not enough overlapping years (need ≥ 8)."), auto_unbox = TRUE))
  quit(save = "no", status = 0)
}

df <- data.frame(year = years_all) %>%
  dplyr::left_join(pat_df, by = "year") %>%
  dplyr::left_join(pub_df, by = "year") %>%
  dplyr::mutate(
    patent_count = ifelse(is.na(patent_count), 0L, patent_count),
    pub_count    = ifelse(is.na(pub_count),    0L, pub_count)
  ) %>%
  dplyr::arrange(year)

# linear exog: pub_lag1
df <- df %>%
  dplyr::mutate(pub_lag1 = dplyr::lag(pub_count, 1)) %>%
  dplyr::filter(!is.na(pub_lag1))

if (nrow(df) < 6) {
  cat(toJSON(list(ok = FALSE, stage = "features", error = "Too few rows after lag alignment."), auto_unbox = TRUE))
  quit(save = "no", status = 0)
}

y_all   <- df$patent_count
x_train <- as.matrix(df$pub_lag1)
yrs     <- df$year

res <- list(ok = TRUE)
res$truncation <- list(patents_drop_last = 3L, pubs_drop_last = 1L)
res$original_data <- list(
  years        = as.integer(yrs),
  patent_count = as.integer(y_all),
  pub_count    = as.integer(df$pub_count)
)

if (mode == "test") {
  # ---- last 5 years as test ----
  k_test <- 5L
  n <- length(yrs)
  if (n <= k_test) {
    cat(toJSON(list(ok = FALSE, stage = "split", error = "Not enough rows to hold out 5 years."), auto_unbox = TRUE))
    quit(save = "no", status = 0)
  }
  train_idx <- seq_len(n - k_test)
  test_idx  <- (n - k_test + 1):n

  y_tr <- y_all[train_idx]; X_tr <- x_train[train_idx, , drop = FALSE]
  y_te <- y_all[test_idx];  X_te <- x_train[test_idx,  , drop = FALSE]
  yrs_tr <- yrs[train_idx]; yrs_te <- yrs[test_idx]

  mod <- forecast::auto.arima(
    y_tr, xreg = X_tr,
    stepwise = FALSE, approximation = FALSE
  )
  fitted_in <- as.numeric(fitted(mod))
  fc <- forecast::forecast(mod, xreg = X_te, h = length(y_te))
  yhat <- as.numeric(fc$mean)
  lo95 <- as.numeric(fc$lower[, 2])
  hi95 <- as.numeric(fc$upper[, 2])

  res$model   <- "arimax-linear"
  res$order   <- as.integer(mod$arma[c(1,6,2)])  # p,d,q
  res$train   <- list(years = as.integer(yrs_tr), fitted = fitted_in)
  res$test    <- list(years = as.integer(yrs_te), yhat = yhat, yhat_lower = lo95, yhat_upper = hi95)
  res$metrics <- list(MAE = as.numeric(mae(y_te, yhat)), RMSE = as.numeric(rmse(y_te, yhat)))

} else if (mode == "future") {
  if (is.na(horizon) || horizon <= 0) {
    cat(toJSON(list(ok = FALSE, stage = "future_args", error = "horizon must be positive."), auto_unbox = TRUE))
    quit(save = "no", status = 0)
  }

  # future pubs: explicit or scenario
  fut_years <- trimws(unlist(strsplit(pub_future_years, ",")))
  fut_vals  <- trimws(unlist(strsplit(pub_future_values, ",")))
  fut_years <- fut_years[nzchar(fut_years)]
  fut_vals  <- fut_vals[nzchar(fut_vals)]
  if (length(fut_years) == horizon && length(fut_vals) == horizon) {
    fut_years <- as.integer(safe_int(fut_years))
    fut_pubs  <- as.numeric(safe_num(fut_vals))
  } else {
    scen <- linear_extrapolate(yrs, df$pub_count, horizon)
    if (pub_strategy == "flat") {
      fut_years <- seq(max(yrs) + 1L, max(yrs) + horizon, by = 1L)
      fut_pubs  <- rep(tail(df$pub_count, 1), horizon)
    } else {
      fut_years <- scen$years
      fut_pubs  <- scen$values
    }
  }

  # fit model on all available (after truncation)
  mod <- forecast::auto.arima(
    y_all, xreg = x_train,
    stepwise = FALSE, approximation = FALSE
  )

  # build future lag-1 path (pub_{t-1})
  last_hist_pub <- tail(df$pub_count, 1)
  lag1 <- numeric(horizon); prev <- as.numeric(last_hist_pub)
  for (i in seq_len(horizon)) { lag1[i] <- prev; prev <- fut_pubs[i] }
  X_future <- as.matrix(lag1)

  fc <- forecast::forecast(mod, xreg = X_future, h = horizon)
  yhat <- as.numeric(fc$mean)
  lo95 <- as.numeric(fc$lower[, 2])
  hi95 <- as.numeric(fc$upper[, 2])

  res$model    <- "arimax-linear"
  res$order    <- as.integer(mod$arma[c(1,6,2)])  # p,d,q
  res$forecast <- list(
    years = as.integer(fut_years),
    yhat = yhat, yhat_lower = lo95, yhat_upper = hi95
  )
} else {
  cat(toJSON(list(ok = FALSE, stage = "mode", error = "mode must be 'test' or 'future'."), auto_unbox = TRUE))
  quit(save = "no", status = 0)
}

cat(toJSON(res, auto_unbox = TRUE, digits = 10))
