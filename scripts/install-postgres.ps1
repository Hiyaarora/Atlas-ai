# ==========================================================================
# Atlas AI - portable PostgreSQL install (no admin, no installer)
#
#   .\scripts\install-postgres.ps1 -ZipPath "$env:USERPROFILE\Downloads\postgresql-17-binaries.zip"
#
# Extracts the official PostgreSQL binaries ZIP and initialises a database
# cluster in your home directory. Nothing is written outside $PG_HOME and
# $PG_DATA, no Windows service is registered, and no elevation is required.
#
# Why not the .exe installer: EnterpriseDB's InstallBuilder installer requires
# elevation and crashes with 0xc0000005 on some Windows 11 machines. The ZIP
# contains the identical binaries with no installer wrapper.
#
# To undo everything this script does, delete the two directories it prints.
# ==========================================================================

param(
    [string]$ZipPath = "$env:USERPROFILE\Downloads\postgresql-17-binaries.zip",
    [string]$PgHome = "$env:USERPROFILE\pgsql",
    [string]$PgData = "$env:USERPROFILE\pgdata",
    [string]$SuperuserPassword = 'postgres'
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $ZipPath)) {
    Write-Error "ZIP not found at $ZipPath"
}

# ---- 1. Extract ----------------------------------------------------------
if (Test-Path (Join-Path $PgHome 'bin\postgres.exe')) {
    Write-Host "Binaries already present at $PgHome - skipping extraction." -ForegroundColor DarkGray
} else {
    Write-Host "Extracting to $PgHome ..." -ForegroundColor Cyan
    $staging = Join-Path ([System.IO.Path]::GetTempPath()) "atlas-pg-$(Get-Random)"
    Expand-Archive -Path $ZipPath -DestinationPath $staging -Force

    # The archive contains a single top-level `pgsql/` directory.
    $inner = Join-Path $staging 'pgsql'
    $source = if (Test-Path $inner) { $inner } else { $staging }

    New-Item -ItemType Directory -Force -Path $PgHome | Out-Null
    Copy-Item -Path (Join-Path $source '*') -Destination $PgHome -Recurse -Force
    Write-Host "Extracted." -ForegroundColor Green
}

$initdb = Join-Path $PgHome 'bin\initdb.exe'
$pgCtl = Join-Path $PgHome 'bin\pg_ctl.exe'
foreach ($tool in $initdb, $pgCtl) {
    if (-not (Test-Path $tool)) { Write-Error "Expected $tool after extraction, but it is missing." }
}

# ---- 2. Initialise the cluster ------------------------------------------
if (Test-Path (Join-Path $PgData 'PG_VERSION')) {
    Write-Host "Data directory already initialised at $PgData - skipping initdb." -ForegroundColor DarkGray
} else {
    Write-Host "`nInitialising cluster at $PgData ..." -ForegroundColor Cyan

    # initdb reads the superuser password from a file so it never appears in
    # the process command line, where any other user could read it.
    $pwFile = Join-Path ([System.IO.Path]::GetTempPath()) "atlas-pgpw-$(Get-Random).txt"
    try {
        Set-Content -Path $pwFile -Value $SuperuserPassword -Encoding ascii -NoNewline

        & $initdb -D $PgData -U postgres --pwfile=$pwFile --encoding=UTF8 --auth=scram-sha-256
        if ($LASTEXITCODE -ne 0) { Write-Error "initdb failed." }
    } finally {
        if (Test-Path $pwFile) { Remove-Item $pwFile -Force }
    }
    Write-Host "Cluster initialised." -ForegroundColor Green
}

# ---- 3. Start ------------------------------------------------------------
$logFile = Join-Path $PgData 'server.log'
& $pgCtl -D $PgData status *> $null
if ($LASTEXITCODE -eq 0) {
    Write-Host "Server already running." -ForegroundColor DarkGray
} else {
    Write-Host "`nStarting server ..." -ForegroundColor Cyan
    & $pgCtl -D $PgData -l $logFile start
    if ($LASTEXITCODE -ne 0) { Write-Error "Server failed to start. See $logFile" }
}

Write-Host @"

PostgreSQL is installed and running.

  binaries : $PgHome
  data     : $PgData
  log      : $logFile
  port     : 5432
  superuser: postgres / $SuperuserPassword

Day-to-day control:
  .\scripts\db.ps1 start | stop | status | logs | psql

Next:
  .\scripts\setup-db.ps1      # create the atlas role and database

To remove PostgreSQL entirely, delete $PgHome and $PgData.
"@ -ForegroundColor Green
