@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM ================ CONFIGURACIÓN (EDITA AQUÍ) ================
REM Ruta del repositorio
set "REPO=C:\Users\chems\WEBPREDICCION\Prophet_TQQQ"

REM Rama a usar. Si la dejas vacía, se detecta automáticamente.
set "BRANCH=main"

REM Limitar los commits solo a la carpeta data\? (sí = data, no = vacío)
REM Ejemplos:
REM   set "SCOPE_PATH=data"
REM   set "SCOPE_PATH="
set "SCOPE_PATH=data"

REM Ruta de git.exe. Si está vacía o no existe, se usa el que esté en PATH.
set "GIT=%ProgramFiles%\Git\cmd\git.exe"

REM Archivo de log
set "LOG=%REPO%\git_push_log.txt"
REM ============================================================


REM ===== Encabezado de log y validaciones básicas =====
if not exist "%REPO%" (
  echo [%date% %time%] ERROR: Repo no encontrado: %REPO%>> "%LOG%"
  echo ERROR: Repo no encontrado: %REPO%
  exit /b 1
)
echo.>> "%LOG%"
echo [%date% %time%] ===== START git_auto_push.bat ===== >> "%LOG%"
echo REPO=%REPO%  BRANCH=%BRANCH%  GIT=%GIT%  SCOPE=%SCOPE_PATH% >> "%LOG%"

cd /d "%REPO%" || (
  echo [%date% %time%] ERROR: cd al repo fallo >> "%LOG%"
  echo ERROR: No se pudo entrar al repositorio.
  exit /b 1
)

REM ===== Resolver git.exe (fallback al PATH si no existe en %ProgramFiles%) =====
if not exist "%GIT%" (
  for /f "usebackq delims=" %%G in (`where git 2^>nul`) do set "GIT=%%G"
)
"%GIT%" --version >> "%LOG%" 2>&1 || (
  echo [%date% %time%] ERROR: git no encontrado >> "%LOG%"
  echo ERROR: Git no encontrado (ni en %ProgramFiles% ni en PATH).
  exit /b 1
)

REM ===== Detectar rama si no está fija =====
if not defined BRANCH (
  for /f "usebackq delims=" %%B in (`"%GIT%" rev-parse --abbrev-ref HEAD 2^>nul`) do set "BRANCH=%%B"
)
echo [%date% %time%] Rama efectiva: %BRANCH% >> "%LOG%"
echo Rama: %BRANCH%
if not defined BRANCH (
  echo [%date% %time%] ERROR: No se pudo detectar la rama actual. >> "%LOG%"
  echo ERROR: No se pudo detectar la rama actual.
  exit /b 1
)

REM ===== Verificar remoto origin =====
"%GIT%" remote get-url origin >> "%LOG%" 2>&1
if errorlevel 1 (
  echo [%date% %time%] ERROR: Remoto 'origin' no configurado. >> "%LOG%"
  echo ERROR: Remoto 'origin' no configurado.
  exit /b 1
)

REM ===== Detectar cambios (incluye untracked). Si SCOPE_PATH está vacío, analiza todo. =====
set "STATUS_CMD=%GIT% status --porcelain"
if defined SCOPE_PATH set "STATUS_CMD=%STATUS_CMD% -- %SCOPE_PATH%"

for /f "usebackq delims=" %%L in (`%STATUS_CMD% 2^>nul`) do (
  set "HAS_CHANGES=1"
  >>"%REPO%\_changed.tmp" echo %%L
)

if not defined HAS_CHANGES (
  echo [%date% %time%] Sin cambios para commitear (scope: %SCOPE_PATH%). >> "%LOG%"
  echo Sin cambios. Nada que hacer.
  goto :END_OK_NOCHANGES
)

echo [%date% %time%] Cambios detectados (scope: %SCOPE_PATH%): >> "%LOG%"
type "%REPO%\_changed.tmp" >> "%LOG%"
echo Cambios detectados. Procediendo...

REM ===== Mensaje con timestamp =====
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HH-mm"') do set "NOW=%%I"
set "MSG=Auto-update %SCOPE_PATH% %NOW%"
if not defined SCOPE_PATH set "MSG=Auto-update repo %NOW%"
echo [%date% %time%] Commit msg: %MSG% >> "%LOG%"

REM ===== git add (limitado a SCOPE_PATH si aplica) =====
if defined SCOPE_PATH (
  "%GIT%" add -A -- "%SCOPE_PATH%" >> "%LOG%" 2>&1
) else (
  "%GIT%" add -A >> "%LOG%" 2>&1
)

REM ===== ¿Quedó algo en staging? (diff --cached --quiet devuelve 1 si hay algo) =====
"%GIT%" diff --cached --quiet
if not errorlevel 1 (
  echo [%date% %time%] Nada quedó en staging tras add (posible .gitignore/filtros). >> "%LOG%"
  echo Nada quedó en staging tras add (posible .gitignore). Saliendo.
  goto :END_OK_NOCHANGES
)

REM ===== Commit =====
"%GIT%" commit -m "%MSG%" >> "%LOG%" 2>&1
if errorlevel 1 (
  echo [%date% %time%] Nada que commitear tras add. >> "%LOG%"
  echo Nada que commitear tras add. Saliendo.
  goto :END_OK_NOCHANGES
)

REM ===== Mostrar resumen del último commit =====
"%GIT%" show --name-status --oneline -1 >> "%LOG%" 2>&1

REM ===== Push =====
"%GIT%" push -u origin %BRANCH% >> "%LOG%" 2>&1
if errorlevel 1 (
  echo [%date% %time%] ERROR en push a %BRANCH%. >> "%LOG%"
  "%GIT%" remote -v >> "%LOG%"
  "%GIT%" branch -vv >> "%LOG%"
  echo ERROR en push. Revisa el log: %LOG%
  exit /b 1
)

echo [%date% %time%] Push OK a %BRANCH%. >> "%LOG%"
echo ✅ Commit y push OK a %BRANCH%.
echo Log: %LOG%
goto :END_OK

:END_OK_NOCHANGES
echo [%date% %time%] ===== END (no changes) ===== >> "%LOG%"
del "%REPO%\_changed.tmp" >nul 2>&1
endlocal
exit /b 0

:END_OK
echo [%date% %time%] ===== END (ok) ===== >> "%LOG%"
del "%REPO%\_changed.tmp" >nul 2>&1
endlocal
exit /b 0
