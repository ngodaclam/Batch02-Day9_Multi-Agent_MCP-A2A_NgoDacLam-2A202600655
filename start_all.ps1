# PowerShell script to start all services on Windows (Supervisor-Workers Pattern)
# Runs each service in a separate CMD window for easy log viewing and debugging.

$registry = Start-Process cmd -ArgumentList "/k uv run python -m registry" -PassThru
Write-Host "Registry started on port 10000."
Start-Sleep -Seconds 2

$legal = Start-Process cmd -ArgumentList "/k uv run python -m legal_worker" -PassThru
Write-Host "Legal Analysis Worker started on port 10104."

$tax = Start-Process cmd -ArgumentList "/k uv run python -m tax_agent" -PassThru
Write-Host "Tax Worker started on port 10102."

$compliance = Start-Process cmd -ArgumentList "/k uv run python -m compliance_agent" -PassThru
Write-Host "Compliance Worker started on port 10103."
Start-Sleep -Seconds 2

$supervisor = Start-Process cmd -ArgumentList "/k uv run python -m law_agent" -PassThru
Write-Host "Supervisor Agent started on port 10101."
Start-Sleep -Seconds 2

$customer = Start-Process cmd -ArgumentList "/k uv run python -m customer_agent" -PassThru
Write-Host "Customer Agent started on port 10100."

Write-Host ""
Write-Host "All services started (Supervisor-Workers Pattern):"
Write-Host "  Registry:               http://localhost:10000"
Write-Host "  Customer Agent:         http://localhost:10100"
Write-Host "  Supervisor Agent:       http://localhost:10101  (orchestrator)"
Write-Host "  Tax Worker:             http://localhost:10102  (worker)"
Write-Host "  Compliance Worker:      http://localhost:10103  (worker)"
Write-Host "  Legal Analysis Worker:  http://localhost:10104  (worker)"
Write-Host ""
Write-Host "Architecture: Customer -> Supervisor -> [Legal + Tax + Compliance] (parallel)"
Write-Host ""
Write-Host "Press enter in this terminal to stop all of them..."
Read-Host

Write-Host "Stopping all services..."
Stop-Process -Id $customer.Id -Force -ErrorAction SilentlyContinue
Stop-Process -Id $supervisor.Id -Force -ErrorAction SilentlyContinue
Stop-Process -Id $compliance.Id -Force -ErrorAction SilentlyContinue
Stop-Process -Id $tax.Id -Force -ErrorAction SilentlyContinue
Stop-Process -Id $legal.Id -Force -ErrorAction SilentlyContinue
Stop-Process -Id $registry.Id -Force -ErrorAction SilentlyContinue

# Force cleanup orphaned python processes running on project ports
$ports = @(10000, 10100, 10101, 10102, 10103, 10104)
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
