#!/bin/bash
set -e

# Start all Legal Multi-Agent System services (Supervisor-Workers Pattern)
# Registry must be first, then leaf workers, then Supervisor, then Customer Agent

echo "Starting Registry service on port 10000..."
python -m registry &
REGISTRY_PID=$!
sleep 2

echo "Starting Legal Analysis Worker on port 10104..."
python -m legal_worker &
LEGAL_PID=$!

echo "Starting Tax Worker on port 10102..."
python -m tax_agent &
TAX_PID=$!

echo "Starting Compliance Worker on port 10103..."
python -m compliance_agent &
COMPLIANCE_PID=$!
sleep 3

echo "Starting Supervisor Agent on port 10101..."
python -m law_agent &
SUPERVISOR_PID=$!
sleep 3

echo "Starting Customer Agent on port 10100..."
python -m customer_agent &
CUSTOMER_PID=$!

echo ""
echo "All services started (Supervisor-Workers Pattern):"
echo "  Registry:               http://localhost:10000"
echo "  Customer Agent:         http://localhost:10100"
echo "  Supervisor Agent:       http://localhost:10101  (orchestrator)"
echo "  Tax Worker:             http://localhost:10102  (worker)"
echo "  Compliance Worker:      http://localhost:10103  (worker)"
echo "  Legal Analysis Worker:  http://localhost:10104  (worker)"
echo ""
echo "Architecture: Customer → Supervisor → [Legal + Tax + Compliance] (parallel)"
echo ""
echo "Run test_client.py to send a query:"
echo "  python test_client.py"
echo ""
echo "Press Ctrl+C to stop all services."

# Wait for all background processes
wait $REGISTRY_PID $LEGAL_PID $TAX_PID $COMPLIANCE_PID $SUPERVISOR_PID $CUSTOMER_PID