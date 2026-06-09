$ports = @(10000, 10100, 10101, 10102, 10103, 10104)
foreach ($port in $ports) {
    $conn = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($conn) {
        $procIds = $conn.OwningProcess | Select-Object -Unique
        foreach ($procId in $procIds) {
            Write-Host "Killing PID $procId on port $port"
            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        }
    }
}
Write-Host "All ports cleaned."
