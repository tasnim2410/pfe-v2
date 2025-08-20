# r_scripts/tvp_var_forecast.R
# -------------------------------------------------------------
# Reads DB params and options from commandArgs, runs your TVP-VAR,
# and prints JSON to stdout (cat(toJSON(...))) so Flask can return it.
# -------------------------------------------------------------

suppressPackageStartupMessages({
  library(DBI)
  library(RPostgres)
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(ggplot2)
  library(tvReg)
  library(jsonlite)
})

# ---- Parse command line args --------------------------------
args <- commandArgs(trailingOnly = TRUE)
kv <- strsplit(args, "=", fixed = TRUE)
opts <- setNames(lapply(kv, `[`, 2), vapply(kv, `[`, "", 1))
get_opt <- function(key, default = NULL) if (!is.null(opts[[key]])) opts[[key]] else default

db_params <- list(
  dbname   = get_opt("dbname",   "patent_db"),
  host     = get_opt("host",     "localhost"),
  port     = as.integer(get_opt("port", "5433")),
  user     = get_opt("user",     "postgres"),
  password = get_opt("password", "tasnim")
)

truncate_last_n <- as.integer(get_opt("truncate_last_n", "4"))
forecast_h      <- as.integer(get_opt("forecast_h", "10"))
lags_csv        <- get_opt("lags", "1,2,3")
lags_to_test    <- as.integer(strsplit(lags_csv, ",", fixed = TRUE)[[1]])

# ---- Connect to Database ------------------------------------
con <- tryCatch(
  dbConnect(
    RPostgres::Postgres(),
    dbname   = db_params$dbname,
    host     = db_params$host,
    port     = db_params$port,
    user     = db_params$user,
    password = db_params$password
  ),
  error = function(e) {
    stop("Database connection failed: ", conditionMessage(e))
  }
)
on.exit({ try(dbDisconnect(con), silent = TRUE) }, add = TRUE)

# ---- Fetch Data ---------------------------------------------
query_patents <- "
  SELECT first_filing_year AS year, COUNT(*) AS patent_count
  FROM raw_patents
  GROUP BY first_filing_year
  ORDER BY first_filing_year;
"
query_pubs <- "
  SELECT year, COUNT(*) AS pub_count
  FROM research_data3
  GROUP BY year
  ORDER BY year;
"

patents_df <- dbGetQuery(con, query_patents)
pubs_df    <- dbGetQuery(con, query_pubs)
if (nrow(patents_df) == 0L) stop("No rows returned from raw_patents.")
if (nrow(pubs_df)    == 0L) stop("No rows returned from research_data3.")

# ---- Merge/Clean --------------------------------------------
df <- dplyr::full_join(patents_df, pubs_df, by = "year") |>
  dplyr::mutate(year = as.integer(year)) |>
  dplyr::arrange(year) |>
  tidyr::complete(
    year = seq.int(min(year, na.rm = TRUE), max(year, na.rm = TRUE), 1L),
    fill = list(patent_count = 0, pub_count = 0)
  ) |>
  dplyr::mutate(
    patent_count = as.numeric(patent_count),
    pub_count    = as.numeric(pub_count)
  )

# Truncate last N years
last_complete_year <- max(df$year, na.rm = TRUE) - truncate_last_n
df <- dplyr::filter(df, year <= last_complete_year)
if (nrow(df) < 12L) stop("Not enough years after truncation for TVP-VAR.")

# ---- Log-diff + helpers -------------------------------------
start_year <- min(df$year); end_year <- max(df$year)
y_log <- log1p(df$patent_count)
x_log <- log1p(df$pub_count)
d_patents <- c(NA_real_, diff(y_log))
d_pubs    <- c(NA_real_, diff(x_log))

train_df <- tibble::tibble(
  t          = seq_along(d_patents)[-1],
  year       = df$year[-1],
  d_patents  = d_patents[-1],
  d_pubs     = d_pubs[-1]
)

build_tvp_lag_df <- function(obj, p) {
  stopifnot(all(c("d_patents", "d_pubs") %in% names(obj)))
  base <- tibble::tibble(
    t         = seq_len(nrow(obj)),
    d_patents = as.numeric(obj[["d_patents"]]),
    d_pubs    = as.numeric(obj[["d_pubs"]])
  )
  for (k in seq_len(p)) {
    base[[paste0("d_patents_l", k)]] <- dplyr::lag(base$d_patents, k)
    base[[paste0("d_pubs_l", k)]]    <- dplyr::lag(base$d_pubs,    k)
  }
  dplyr::filter(
    base,
    !dplyr::if_any(dplyr::starts_with("d_patents_l"), is.na),
    !dplyr::if_any(dplyr::starts_with("d_pubs_l"),    is.na)
  )
}

tvp_var_forecast <- function(p, df_in = train_df, h = forecast_h) {
  tv_df  <- build_tvp_lag_df(df_in, p)
  x_cols <- c(paste0("d_patents_l", seq_len(p)), paste0("d_pubs_l", seq_len(p)))

  dat1 <- dplyr::select(tv_df, y = d_patents, dplyr::all_of(x_cols))
  dat2 <- dplyr::select(tv_df, y = d_pubs,    dplyr::all_of(x_cols))

  fit1 <- tvReg::tvLM(y ~ ., data = dat1, z = tv_df$t)
  fit2 <- tvReg::tvLM(y ~ ., data = dat2, z = tv_df$t)

  # Get fitted values for historical data
  fitted_pat <- as.numeric(stats::fitted(fit1))
  fitted_pub <- as.numeric(stats::fitted(fit2))
  
  # Reconstruct historical levels from fitted differences
  hist_years <- df_in$year[tv_df$t]
  hist_pat_levels <- numeric(length(fitted_pat))
  hist_pub_levels <- numeric(length(fitted_pub))
  
  # Find the starting levels for reconstruction
  start_idx <- tv_df$t[1]
  hist_pat_levels[1] <- exp(y_log[start_idx])
  hist_pub_levels[1] <- exp(x_log[start_idx])
  
  for (i in 2:length(fitted_pat)) {
    hist_pat_levels[i] <- pmax(hist_pat_levels[i-1] * exp(fitted_pat[i]), 0)
    hist_pub_levels[i] <- pmax(hist_pub_levels[i-1] * exp(fitted_pub[i]), 0)
  }

  # Forecast future values
  lag_pat <- rev(utils::tail(tv_df$d_patents, p))
  lag_pub <- rev(utils::tail(tv_df$d_pubs, p))

  d_hat_pat <- numeric(h)
  d_hat_pub <- numeric(h)
  newz_fix <- as.numeric(max(tv_df$t))

  for (s in seq_len(h)) {
    vals  <- stats::setNames(c(lag_pat, lag_pub), x_cols)
    x_new <- matrix(as.numeric(vals[x_cols]), nrow = 1)
    colnames(x_new) <- x_cols
    z_vec <- rep(newz_fix + s, nrow(x_new))
    d_hat_pat[s] <- as.numeric(stats::predict(fit1, newdata = x_new, newz = z_vec))
    d_hat_pub[s] <- as.numeric(stats::predict(fit2, newdata = x_new, newz = z_vec))
    if (p > 1) {
      lag_pat <- c(d_hat_pat[s], lag_pat[1:(p - 1)])
      lag_pub <- c(d_hat_pub[s], lag_pub[1:(p - 1)])
    } else {
      lag_pat <- d_hat_pat[s]
      lag_pub <- d_hat_pub[s]
    }
  }

  last_y <- tail(y_log, 1)
  last_x <- tail(x_log, 1)
  
  forecast_years <- seq(end_year + 1L, by = 1L, length.out = h)
  forecast_pat <- pmax(expm1(last_y + cumsum(d_hat_pat)), 0)
  forecast_pub <- pmax(expm1(last_x + cumsum(d_hat_pub)), 0)

  list(
    p = p,
    historical = list(
      years = hist_years,
      patent_count = hist_pat_levels,
      pub_count = hist_pub_levels
    ),
    forecast = list(
      years = forecast_years,
      patent_count = forecast_pat,
      pub_count = forecast_pub
    )
  )
}

# ---- Run and output JSON ------------------------------------
if (length(lags_to_test) == 0) stop("No lags to test specified.")

results <- list()
for (p in lags_to_test) {
  res <- tryCatch(
    tvp_var_forecast(p),
    error = function(e) {
      message(sprintf("tvp_var_forecast failed for lag %s: %s", p, conditionMessage(e)))
      NULL
    }
  )
  if (!is.null(res)) {
    results[[paste0("p", p)]] <- res
  }
}

if (length(results) == 0) stop("No successful TVP-VAR results for any lag.")

# Create comprehensive output structure
output_list <- list(
  models = results,
  original_data = list(
    years = df$year,
    patent_count = df$patent_count,
    pub_count = df$pub_count
  )
)

cat(jsonlite::toJSON(output_list, auto_unbox = TRUE))
