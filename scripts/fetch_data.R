# --- Bootstrap para ejecutar desde Task Scheduler ---
.libPaths(c("C:/Users/chems/AppData/Local/R/win-library/4.5", .libPaths()))
options(repos = "https://cloud.r-project.org")

# Instala paquetes faltantes solo si es necesario
pkgs <- c("quantmod", "xts", "TTR", "PerformanceAnalytics")
to_install <- setdiff(pkgs, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install, dependencies = TRUE)

# Carga paquetes
invisible(lapply(pkgs, require, character.only = TRUE))

# --- Lógica de descarga ---
fecha_final <- Sys.Date()

descargar_y_guardar <- function(ticker, nombre_archivo) {
  datos <- getSymbols(ticker, from = "2020-01-01", to = fecha_final, src = "yahoo", auto.assign = FALSE)
  df <- data.frame(Date = index(datos), coredata(datos))
  write.csv(df, file = paste0("C:/Users/chems/WEBPREDICCION/Prophet_TQQQ/data/", nombre_archivo), row.names = FALSE)
}

tickers <- list(
  TQQQ = "TQQQ_data.csv",
  QQQ = "QQQ_data.csv",
  UPRO = "UPRO_data.csv",
  SPY = "SPY_data.csv",
  SOXL = "SOXL_data.csv",
  RHHBY = "RHHBY_data.csv",
  XLV = "XLV_data.csv",
  IXJ = "IXJ_data.csv",
  `BTC-USD` = "BTC_data.csv",
  IBIT = "IBIT_data.csv",
  TSLA = "TSLA_data.csv",
  TSLG = "TSLG_data.csv",
  DIA = "DIA_data.csv",
  UDOW = "UDOW_data.csv"
)

for (i in names(tickers)) {
  descargar_y_guardar(i, tickers[[i]])
}

# --- Registro ---
cat("✓ Descarga completada hasta:", Sys.Date(), "\n")
cat(paste0("Ejecución: ", Sys.time(), "\n"),
    file = "C:/Users/chems/WEBPREDICCION/Prophet_TQQQ/data/registro_ejecucion.txt",
    append = TRUE)

