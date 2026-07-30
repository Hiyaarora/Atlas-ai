# ==========================================================================
# Atlas AI - create the local PostgreSQL role and database
#
#   .\scripts\setup-db.ps1
#
# Reads POSTGRES_* from the repo-root .env and creates a matching role and
# database. Safe to re-run: it skips anything that already exists.
#
# You will be prompted for the password of the `postgres` superuser - the one
# you chose during the PostgreSQL installer. That is NOT the same as
# POSTGRES_PASSWORD in .env, which belongs to the app's own `atlas` role.
#
# Why a dedicated role instead of just using `postgres`: the application
# should own exactly one database and have no authority over the rest of the
# server. That is the same principle you will apply in production, practised
# from day one.
# ==========================================================================

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $repoRoot '.env'

if (-not (Test-Path $envFile)) {
    Write-Error "No .env found at $envFile. Copy .env.example to .env first."
}

# ---- Parse .env ----------------------------------------------------------
$config = @{}
foreach ($line in Get-Content $envFile) {
    if ($line -match '^\s*#' -or $line -notmatch '=') { continue }
    $key, $value = $line -split '=', 2
    $key = $key.Trim()
    $value = $value.Trim()
    # Strip an inline comment, then surrounding quotes.
    if ($value -match '^(.*?)\s+#') { $value = $matches[1] }
    $config[$key] = $value.Trim().Trim('"').Trim("'")
}

$dbUser = $config['POSTGRES_USER']
$dbPass = $config['POSTGRES_PASSWORD']
$dbName = $config['POSTGRES_DB']
$dbPort = $config['POSTGRES_PORT']

foreach ($required in 'POSTGRES_USER', 'POSTGRES_PASSWORD', 'POSTGRES_DB', 'POSTGRES_PORT') {
    if (-not $config[$required]) { Write-Error "$required is missing from .env" }
}

# ---- Locate psql ---------------------------------------------------------
# Checked in order: PATH, an explicit $PG_HOME, the portable install created by
# install-postgres.ps1, then a standard EDB installer location.
$psql = (Get-Command psql -ErrorAction SilentlyContinue).Source

if (-not $psql -and $env:PG_HOME) {
    $candidate = Join-Path $env:PG_HOME 'bin\psql.exe'
    if (Test-Path $candidate) { $psql = $candidate }
}
if (-not $psql) {
    $candidate = Join-Path $env:USERPROFILE 'pgsql\bin\psql.exe'
    if (Test-Path $candidate) { $psql = $candidate }
}
if (-not $psql) {
    $candidate = Get-ChildItem 'C:\Program Files\PostgreSQL\*\bin\psql.exe' -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending | Select-Object -First 1
    if ($candidate) { $psql = $candidate.FullName }
}
if (-not $psql) {
    Write-Error "psql.exe not found. Run .\scripts\install-postgres.ps1, or add PostgreSQL's bin\ folder to PATH."
}

Write-Host "Using $psql" -ForegroundColor DarkGray
Write-Host "Creating role '$dbUser' and database '$dbName' on port $dbPort" -ForegroundColor Cyan
Write-Host "You will be prompted for the 'postgres' superuser password.`n" -ForegroundColor Yellow

# ---- Create role and database -------------------------------------------
# Escape single quotes in the password so the SQL literal stays valid.
$escapedPass = $dbPass -replace "'", "''"

$sql = @"
DO
`$`$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '$dbUser') THEN
        CREATE ROLE $dbUser WITH LOGIN PASSWORD '$escapedPass';
        RAISE NOTICE 'role % created', '$dbUser';
    ELSE
        ALTER ROLE $dbUser WITH LOGIN PASSWORD '$escapedPass';
        RAISE NOTICE 'role % already existed; password synced with .env', '$dbUser';
    END IF;
END
`$`$;
"@

$sql | & $psql -U postgres -h localhost -p $dbPort -v ON_ERROR_STOP=1 -d postgres
if ($LASTEXITCODE -ne 0) { Write-Error "Failed to create the role." }

# CREATE DATABASE cannot run inside a DO block, so check first.
$exists = & $psql -U postgres -h localhost -p $dbPort -tAc `
    "SELECT 1 FROM pg_database WHERE datname = '$dbName'" -d postgres

if ($exists -eq '1') {
    Write-Host "database '$dbName' already exists - skipping" -ForegroundColor DarkGray
} else {
    & $psql -U postgres -h localhost -p $dbPort -v ON_ERROR_STOP=1 -d postgres `
        -c "CREATE DATABASE $dbName OWNER $dbUser"
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed to create the database." }
    Write-Host "database '$dbName' created" -ForegroundColor Green
}

# ---- Verify the app's own credentials actually work ----------------------
# Creating things as superuser proves nothing about whether the app can log
# in. Connect as the app role, the way the app will.
$env:PGPASSWORD = $dbPass
& $psql -U $dbUser -h localhost -p $dbPort -d $dbName -tAc "SELECT 'connection ok'" | Out-Null
$connectOk = ($LASTEXITCODE -eq 0)
Remove-Item Env:\PGPASSWORD

if ($connectOk) {
    Write-Host "`nVerified: '$dbUser' can connect to '$dbName'." -ForegroundColor Green
    Write-Host "Next: cd backend; alembic upgrade head" -ForegroundColor Cyan
} else {
    Write-Error "Role and database exist, but '$dbUser' could not connect. Check POSTGRES_PASSWORD in .env."
}
