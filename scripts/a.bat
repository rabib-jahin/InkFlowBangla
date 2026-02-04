@echo off
REM ========================================
REM Safe Training Launcher for RTX 4090
REM ========================================

echo.
echo ====================================================
echo    SAFE TRAINING LAUNCHER
echo ====================================================
echo.

REM Set power limit (requires Admin rights)
echo Setting GPU power limit to 350W...
nvidia-smi -pl 350
if %errorlevel% neq 0 (
    echo.
    echo WARNING: Could not set power limit!
    echo Please run this script as Administrator or run manually:
    echo   nvidia-smi -pl 350
    echo.
    pause
)

echo.
echo ====================================================
echo Starting training with safety features:
echo   - Cooling breaks every 3 epochs
echo   - 15 second cooling duration
echo   - Power limited to 350W
echo   - Batch size 32 (stable)
echo ====================================================
echo.

REM Start training
python bangla_train.py ^
    --pickle_path ./BN-UNIFIED-NO-SINGLE.pickle ^
    --batch_size 32 ^
    --epochs 1000 ^
    --cooling_interval 3 ^
    --cooling_duration 15 ^
    --power_limit 350 ^
    --num_workers 0 ^                       
    --save_path ./models ^
    --stable_dif_path runwayml/stable-diffusion-v1-5 ^
    --load_check True ^
    --wandb_log True ^
    --model_name diffusionpen ^
    --style_path ./style_models/mixed_bengali_mobilenetv2_100.pth

echo.
echo ====================================================
echo Training completed or stopped!
echo ====================================================
pause