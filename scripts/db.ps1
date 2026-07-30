# ==========================================================================
# Atlas AI - local PostgreSQL control
#
#   .\scripts\db.ps1 start     # start the server
#   .\scripts\db.ps1 stop      # stop it
#   .\scripts\db.ps1 status    # is it running?
#   .\scripts\db.ps1 logs      # tail the server log
#   .\scripts\db.ps1 psql      # open a SQL shell on the atlas database
#
# This drives a *portable* PostgreSQL install: binaries extracted from the
# official ZIP, no Windows service, no admin rights. The tradeoff is that the
# server does not start with Windows - run `db.ps1 start` when you sit down to
# work. A container would handle its own lifecycle.
#
# Override the locations with environment variables if you installed
# elsewhere:  $env:PG_HOME, $env:PG_DATA
# ==========================================================================

param(
    [Parameter(Position = 0)]
    [ValidateSet('start', 'stop', 'status', 'logs', 'psql')]
    [string]$Command = 'status'
)

$ErrorActionPreference = 'Stop'

$pgHome = if ($env:PG_HOME) { $env:PG_HOME } else { "$env:USERPROFILE\pgsql" }
$pgData = if ($env:PG_DATA) { $env:PG_DATA } else { "$env:USERPROFILE\pgdata" }
$logFile = Join-Path $pgData 'server.log'

$pgCtl = Join-Path $pgHome 'bin\pg_ctl.exe'
$psqlExe = Join-Path $pgHome 'bin\psql.exe'

if (-not (Test-Path $pgCtl)) {
    Write-Error "PostgreSQL binaries not found at $pgHome. Set `$env:PG_HOME or re-run the install."
}
if (-not (Test-Path $pgData)) {
    Write-Error "No data directory at $pgData. Run initdb first (see README)."
}

function Test-Running {
    & $pgCtl -D $pgData status *> $null
    # pg_ctl status: exit 0 = running, 3 = stopped, 4 = no/invalid data dir.
    return ($LASTEXITCODE -eq 0)
}

switch ($Command) {
    'start' {
        if (Test-Running) {
            Write-Host "PostgreSQL is already running." -ForegroundColor DarkGray
            break
        }
        & $pgCtl -D $pgData -l $logFile start
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to start. Check the log: .\scripts\db.ps1 logs"
        }
        Write-Host "PostgreSQL started on port 5432." -ForegroundColor Green
    }

    'stop' {
        if (-not (Test-Running)) {
            Write-Host "PostgreSQL is not running." -ForegroundColor DarkGray
            break
        }
        # -m fast: roll back open transactions and shut down promptly, rather
        # than waiting for every client to disconnect on its own.
        & $pgCtl -D $pgData -m fast stop
        Write-Host "PostgreSQL stopped." -ForegroundColor Yellow
    }

    'status' {
        if (Test-Running) {
            Write-Host "PostgreSQL is RUNNING" -ForegroundColor Green
            Write-Host "  binaries : $pgHome"
            Write-Host "  data     : $pgData"
        } else {
            Write-Host "PostgreSQL is STOPPED" -ForegroundColor Yellow
            Write-Host "  start it with: .\scripts\db.ps1 start"
        }
    }

    'logs' {
        if (-not (Test-Path $logFile)) { Write-Error "No log file at $logFile" }
        Get-Content $logFile -Tail 40 -Wait
    }

    'psql' {
        $envFile = Join-Path (Split-Path -Parent $PSScriptRoot) '.env'
        if (-not (Test-Path $envFile)) { Write-Error "No .env at $envFile" }

        $config = @{}
        foreach ($line in Get-Content $envFile) {
            if ($line -match '^\s*#' -or $line -notmatch '=') { continue }
            $key, $value = $line -split '=', 2
            if ($value -match '^(.*?)\s+#') { $value = $matches[1] }
            $config[$key.Trim()] = $value.Trim().Trim('"').Trim("'")
        }

        $env:PGPASSWORD = $config['POSTGRES_PASSWORD']
        & $psqlExe -U $config['POSTGRES_USER'] -h localhost -p $config['POSTGRES_PORT'] -d $config['POSTGRES_DB']
        Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
    }
}
