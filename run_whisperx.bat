@echo off
echo ========================================
echo HIGH-QUALITY WHISPERX TRANSCRIPTION
echo ========================================
echo.

REM Activate virtual environment
if exist "transcribe\Scripts\activate.bat" (
    echo Activating virtual environment...
    call transcribe\Scripts\activate.bat
) else (
    echo Virtual environment not found. Using system Python.
)

REM Check if audio file is provided
if "%~1"=="" (
    echo Usage: run_whisperx.bat audio_file [output_file] [language]
    echo Example: run_whisperx.bat audio.wav output.srt ro
    echo.
    echo Available languages: en, ro, es, fr, de, it, etc.
    pause
    exit /b 1
)

REM Set arguments
set AUDIO_FILE=%1
set OUTPUT_FILE=%2
set LANGUAGE=%3

REM Default output file if not provided
if "%OUTPUT_FILE%"=="" set OUTPUT_FILE=whisperx_output.srt

REM Run WhisperX transcription
echo Running WhisperX transcription...
echo Input: %AUDIO_FILE%
echo Output: %OUTPUT_FILE%
if not "%LANGUAGE%"=="" echo Language: %LANGUAGE%
echo.

if not "%LANGUAGE%"=="" (
    python transcribe_whisperx.py "%AUDIO_FILE%" "%OUTPUT_FILE%" --language %LANGUAGE%
) else (
    python transcribe_whisperx.py "%AUDIO_FILE%" "%OUTPUT_FILE%"
)

echo.
echo ========================================
echo Transcription complete!
echo Check output file: %OUTPUT_FILE%
echo ========================================
pause 