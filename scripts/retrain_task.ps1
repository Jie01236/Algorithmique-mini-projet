<#
    Enregistre une tache planifiee Windows qui reentraine le modele
    SocialMetrics AI chaque semaine (lundi a 03h00).

    Equivalent Windows du cronjob decrit dans
    scripts/reentrainement_cron.example.

    Utilisation (PowerShell, depuis la racine du projet) :
        ./scripts/retrain_task.ps1

    Pour supprimer la tache :
        Unregister-ScheduledTask -TaskName "SocialMetricsRetrain" -Confirm:$false
#>

$ErrorActionPreference = "Stop"

# Racine du projet (dossier parent de scripts/)
$ProjectDir = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $ProjectDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# Interpreteur Python (venv si present)
$VenvPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $Python = $VenvPython
} else {
    $Python = "python"
}

# Commande executee par la tache : entrainement + log horodate
$Command = "cd `"$ProjectDir`"; & `"$Python`" scripts/train.py *>> `"$LogDir\retrain_task.log`""

$Action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -Command `"$Command`""

# Chaque lundi a 03h00
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At 3am

$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable

Register-ScheduledTask -TaskName "SocialMetricsRetrain" `
    -Action $Action -Trigger $Trigger -Settings $Settings `
    -Description "Reentrainement hebdomadaire du modele SocialMetrics AI" `
    -Force

Write-Host "Tache planifiee 'SocialMetricsRetrain' enregistree (lundi 03h00)."
