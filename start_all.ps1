# PowerShell script to start all services on Windows
# Runs each service in a separate CMD window for easy log viewing and debugging.

$registry = Start-Process cmd -ArgumentList "/k uv run python -m registry" -PassThru
Write-Host "Registry started on port 10000."
Start-Sleep -Seconds 2

$tax = Start-Process cmd -ArgumentList "/k uv run python -m tax_agent" -PassThru
Write-Host "Tax Agent started on port 10102."

$compliance = Start-Process cmd -ArgumentList "/k uv run python -m compliance_agent" -PassThru
Write-Host "Compliance Agent started on port 10103."
Start-Sleep -Seconds 2

$law = Start-Process cmd -ArgumentList "/k uv run python -m law_agent" -PassThru
Write-Host "Law Agent started on port 10101."
Start-Sleep -Seconds 2

$customer = Start-Process cmd -ArgumentList "/k uv run python -m customer_agent" -PassThru
Write-Host "Customer Agent started on port 10100."

Write-Host ""
Write-Host "All services started in separate CMD windows."
Write-Host "Press enter in this terminal to stop all of them..."
Read-Host

Write-Host "Stopping all services..."
Stop-Process -Id $customer.Id -Force -ErrorAction SilentlyContinue
Stop-Process -Id $law.Id -Force -ErrorAction SilentlyContinue
Stop-Process -Id $compliance.Id -Force -ErrorAction SilentlyContinue
Stop-Process -Id $tax.Id -Force -ErrorAction SilentlyContinue
Stop-Process -Id $registry.Id -Force -ErrorAction SilentlyContinue

# Force cleanup orphaned python processes running on project ports
$ports = @(10000, 10100, 10101, 10102, 10103)
foreach ($port in $ports) {
    $conn = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($conn) {
        $pids = $conn.OwningProcess | Select-Object -Unique
        foreach ($pid in $pids) {
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
        }
    }
}

Write-Host "All services stopped."
