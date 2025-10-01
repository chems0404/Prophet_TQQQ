# run_r.ps1 — Ejecuta el Rscript y deja log
param(
  [string]$Repo    = "C:\Users\chems\WEBPREDICCION\Prophet_TQQQ",
  [string]$RScript = "C:\Users\chems\WEBPREDICCION\Prophet_TQQQ\scripts\fetch_data.R",
  # Ajusta según tu versión de R instalada
  [string]$RExec = "C:\Program Files\R\R-4.5.1\bin\Rscript.exe"

)

$ErrorActionPreference = "Stop"
$Log = Join-Path $Repo "fetch_data_log.txt"

function Log($m){ "{0:yyyy-MM-dd HH:mm:ss.fff}  {1}" -f (Get-Date),$m | Tee-Object -FilePath $Log -Append }

# fallback si no existe Rscript en la ruta dada
if (-not (Test-Path $RExec)) {
  $r = (Get-Command Rscript.exe -ErrorAction SilentlyContinue)
  if ($r) { $RExec = $r.Source }
}

if (-not (Test-Path $RExec)) { Log "ERROR: No se encontró Rscript.exe"; exit 1 }
if (-not (Test-Path $RScript)) { Log "ERROR: No se encontró el script R: $RScript"; exit 1 }

Set-Location $Repo
Log "===== START run_r.ps1 ====="
Log "RExec=$RExec"
Log "RScript=$RScript"

# Ejecutar R y capturar salida/errores
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $RExec
$psi.Arguments = "`"$RScript`""
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError  = $true
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true
$p = New-Object System.Diagnostics.Process
$p.StartInfo = $psi
[void]$p.Start()

$stdout = $p.StandardOutput.ReadToEnd()
$stderr = $p.StandardError.ReadToEnd()
$p.WaitForExit()

if ($stdout) { Log $stdout.TrimEnd() }
if ($stderr) { Log "STDERR: " + $stderr.TrimEnd() }

if ($p.ExitCode -ne 0) {
  Log "ERROR: Rscript terminó con código $($p.ExitCode)"
  Log "===== END (error) ====="
  exit $p.ExitCode
}

Log "OK: Rscript finalizado."
Log "===== END (ok) ====="
exit 0
