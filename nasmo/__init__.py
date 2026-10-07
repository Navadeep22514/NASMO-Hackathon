"""nasmo - Explainable Multi-Objective Neural Architecture Search
with machine-checkable global optimality certificates."""

__version__ = "1.0.0"

OBJECTIVE_ORDER = ["accuracy_bp", "latency_us", "energy_uj", "peak_mem_bytes", "fairness_bp"]
SENSE = {"accuracy_bp": "max", "latency_us": "min", "energy_uj": "min",
         "peak_mem_bytes": "min", "fairness_bp": "min"}
