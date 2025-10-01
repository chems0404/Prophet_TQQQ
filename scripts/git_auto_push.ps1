<# ------------- git_auto_push.ps1 -------------
   Automático: add/commit/push si hay cambios.
   Configura REPO, SCOPE y BRANCH abajo.
#>

param(
  [string]$Repo   = "C:\Users\chems\WEBPREDICCION\Prophet_TQQQ",
  [string]$Scope  = "data",  # "" para todo el repo; "data" solo esa carpeta
  [string]$Branch = "main"   # "" para autodetectar
)

$ErrorActionPreference = "Stop"
$Log = Join-Path $Repo "git_push_log.txt"

function Log($msg) {
  $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
  "$ts  $msg" | Tee-Object -FilePath $Log -Append
}

# ==== Validaciones básicas
if (-not (Test-Path $Repo)) { Log "ERROR: Repo no encontrado: $Repo"; throw "Repo no encontrado" }
Set-Location $Repo
Log "===== START git_auto_push.ps1 ====="
Log "REPO=$Repo  SCOPE=$Scope  BRANCH=$Branch"

# Git disponible
git --version *> $null

# Detectar rama si no viene fija
if (-not $Branch) {
  $Branch = (git rev-parse --abbrev-ref HEAD).Trim()
  if (-not $Branch) { Log "ERROR: No se pudo detectar la rama actual"; throw "Sin rama" }
}
Log "Rama efectiva: $Branch"

# Confirmar remoto
git remote get-url origin *> $null

# ==== Detectar cambios (incluye untracked)
$statusArgs = @("status","--porcelain")
if ($Scope) { $statusArgs += @("--",$Scope) }
$changes = git @statusArgs

if ([string]::IsNullOrWhiteSpace($changes)) {
  Log "Sin cambios para commitear (scope: $Scope)."
  Log "===== END (no changes) ====="
  exit 0
}

Log "Cambios detectados:`n$changes"
Write-Host "Cambios detectados. Procediendo..."

# ==== Staging
if ($Scope) { git add -A -- $Scope } else { git add -A }

# ¿Quedó algo staged?
git diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
  Log "Nada quedó en staging tras add (posible .gitignore/filtros)."
  Log "===== END (nothing staged) ====="
  exit 0
}

# ==== Commit
$stamp = Get-Date -Format "yyyy-MM-dd_HH-mm"
$msg = if ($Scope) { "Auto-update $Scope $stamp" } else { "Auto-update repo $stamp" }
Log "Commit msg: $msg"

git commit -m $msg *> $null
if ($LASTEXITCODE -ne 0) {
  Log "Nada que commitear tras add."
  Log "===== END (nothing to commit) ====="
  exit 0
}

# Registrar resumen del commit
(git show --name-status --oneline -1) | ForEach-Object { Log $_ }

# ==== Push
git push -u origin $Branch
if ($LASTEXITCODE -ne 0) {
  Log "ERROR en push a $Branch."
  Log (git remote -v | Out-String)
  Log (git branch -vv | Out-String)
  throw "Push falló"
}

Log "Push OK a $Branch."
Log "===== END (ok) ====="
Write-Host "✅ Commit y push OK a $Branch."
Write-Host "Log: $Log"
