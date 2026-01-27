$blender = "C:\Program Files\Blender Foundation\Blender 3.6\blender.exe"
$blendDir = "C:\development\msds_practicum\msds_practicum\data\external"
$script = "C:\development\msds_practicum\msds_practicum\scripts\generate.py"

$blendFiles = Get-ChildItem "$blendDir\t92_*_rot*.blend" | Sort-Object Name

foreach ($blend in $blendFiles) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Processing: $($blend.Name)" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    
    & $blender --background $blend.FullName --python $script
    
    Write-Host "Completed: $($blend.Name)" -ForegroundColor Cyan
    Start-Sleep -Seconds 3
}

Write-Host ""
Write-Host "ALL ELEVATIONS COMPLETE!" -ForegroundColor Green