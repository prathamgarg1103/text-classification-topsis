@echo off
echo ========================================================
echo   TOPSIS Analysis - GitHub Repository Setup (Debug Mode)
echo ========================================================
echo.

echo 1. Searching for Git...
set GIT_PATH=git
where git >nul 2>nul
if %errorlevel% equ 0 goto :FoundGit

echo Git not found in PATH. Checking common locations...
if exist "%LOCALAPPDATA%\GitHubDesktop\app-3.5.4\resources\app\git\cmd\git.exe" (
    set "GIT_PATH=%LOCALAPPDATA%\GitHubDesktop\app-3.5.4\resources\app\git\cmd\git.exe"
    goto :FoundGit
)
if exist "C:\Program Files\Git\cmd\git.exe" (
    set "GIT_PATH=C:\Program Files\Git\cmd\git.exe"
    goto :FoundGit
)
if exist "C:\Program Files\Git\bin\git.exe" (
    set "GIT_PATH=C:\Program Files\Git\bin\git.exe"
    goto :FoundGit
)
if exist "%LOCALAPPDATA%\Programs\Git\cmd\git.exe" (
    set "GIT_PATH=%LOCALAPPDATA%\Programs\Git\cmd\git.exe"
    goto :FoundGit
)

echo [ERROR] Git not found!
echo Please ensure Git is installed. If you just installed it, try restarting your computer or VS Code.
pause
exit /b

:FoundGit
echo [OK] Using Git at: %GIT_PATH%
echo.

echo 2. Initializing git repository...
%GIT_PATH% init
if %errorlevel% neq 0 pause && exit /b
echo.

echo 3. Adding files...
%GIT_PATH% add .
if %errorlevel% neq 0 pause && exit /b
echo.

echo 4. Committing files...
%GIT_PATH% commit -m "Initial commit: Text Classification TOPSIS Analysis"
REM It's okay if this fails if there are no changes to commit
echo.

echo 5. Renaming branch to 'main'...
%GIT_PATH% branch -M main
echo.

echo 6. Setting remote origin...
set repo_url=https://github.com/prathamgarg1103/text-classification-topsis.git
%GIT_PATH% remote remove origin 2>nul
%GIT_PATH% remote add origin %repo_url%
if %errorlevel% neq 0 pause && exit /b
echo.

echo 7. Pushing to GitHub...
echo Pushing to: %repo_url%
%GIT_PATH% push -u origin main
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Push failed. 
    echo Common reasons:
    echo  - Repo doesn't exist (Check URL)
    echo  - Permission denied (You might need to sign in)
    echo.
    pause
    exit /b
)

echo.
echo ========================================================
echo [SUCCESS] Project successfully uploaded to GitHub!
echo ========================================================
pause
