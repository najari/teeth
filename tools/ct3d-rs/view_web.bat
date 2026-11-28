@echo off
rem Builds the web (wasm) version of the CT 3D viewer and opens a GLB file in the browser.
rem
rem   view_web.bat <file.glb> ["view=right&hide=jaw"]
rem   or drag a GLB file onto this file.
rem
rem The optional second argument is added to the page address: view, hide, bone, cut, pick.
rem The GLB (and parts_info.json next to it, if any) is copied to target\web-site, which git does not track,
rem and is served only to this PC (127.0.0.1). Close the server window to stop.
rem The first build takes a few minutes; later runs only rebuild what changed.
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "WB_VERSION=0.2.130"
set "PORT=8000"
set "SITE=%~dp0target\web-site"

if "%~1"=="" (
  echo Usage: view_web.bat file.glb ["view=right&hide=jaw"]
  echo Or drag a GLB file onto this file.
  goto :err
)
if not exist "%~1" (
  echo File not found: "%~1"
  goto :err
)
set "GLB=%~f1"
set "NAME=%~nx1"
set "INFO=%~dp1parts_info.json"
set "EXTRA=%~2"

rem --- tools ---
where cargo >nul 2>nul || (
  echo Rust was not found. Install it from https://rustup.rs and run this again.
  goto :err
)
rustup target list --installed | findstr /x "wasm32-unknown-unknown" >nul || (
  echo [setup] Adding the wasm32-unknown-unknown target...
  rustup target add wasm32-unknown-unknown || goto :err
)
rem wasm-bindgen-cli must match the wasm-bindgen version in Cargo.lock
wasm-bindgen --version 2>nul | findstr /c:"%WB_VERSION%" >nul || (
  echo [setup] Installing wasm-bindgen-cli %WB_VERSION%. This takes a few minutes once.
  cargo install wasm-bindgen-cli --version %WB_VERSION% --locked || goto :err
)
set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY (where py >nul 2>nul && set "PY=py")
if not defined PY (
  echo Python was not found. It is only used as a local web server. Install it from python.org.
  goto :err
)

rem --- build ---
echo [1/4] Building the web version...
cargo build --profile web --target wasm32-unknown-unknown || goto :err
echo [2/4] Generating the JavaScript glue...
if not exist "%SITE%" mkdir "%SITE%"
wasm-bindgen --target web --no-typescript --out-dir "%SITE%" target\wasm32-unknown-unknown\web\ct3d-viewer.wasm || goto :err

rem --- files ---
echo [3/4] Copying the page, the GLB and the font...
copy /y web\index.html "%SITE%\" >nul || goto :err
copy /y "%GLB%" "%SITE%\%NAME%" >nul || goto :err
if exist "%INFO%" (
  copy /y "%INFO%" "%SITE%\parts_info.json" >nul
) else (
  if exist "%SITE%\parts_info.json" del "%SITE%\parts_info.json"
)
rem Browsers do not expose system fonts, so the page loads font.ttf for Korean text.
rem Malgun Gothic is fine on this PC but must not be redistributed; use a free Korean font to share the folder.
if not exist "%SITE%\font.ttf" (
  if exist "%WINDIR%\Fonts\malgun.ttf" (
    copy /y "%WINDIR%\Fonts\malgun.ttf" "%SITE%\font.ttf" >nul
  ) else (
    echo Note: no font.ttf, so Korean text will show as boxes. Put a Korean .ttf named font.ttf in "%SITE%".
  )
)

rem --- serve and open ---
set "URL=http://127.0.0.1:%PORT%/index.html?glb=%NAME%"
if defined EXTRA set "URL=%URL%&%EXTRA%"
echo [4/4] Starting the local server and opening the browser.
echo   Address: "%URL%"
echo   Close the "CT 3D web server" window to stop.
start "CT 3D web server - close this window to stop" /d "%SITE%" %PY% -m http.server %PORT% --bind 127.0.0.1
timeout /t 2 /nobreak >nul
if not defined CT3D_NO_BROWSER start "" "%URL%"
exit /b 0

:err
echo.
echo Stopped. See the messages above.
pause
exit /b 1
