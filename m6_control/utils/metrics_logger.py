class MetricsLogger:
    def __init__(self):
        self.cte_history = []
        self.speed_error_history = []

    def log_step(self, cte, speed_error):
        self.cte_history.append(cte)
        self.speed_error_history.append(speed_error)

    def print_summary(self):
        if not self.cte_history:
            return
        max_cte = max([abs(e) for e in self.cte_history])
        avg_cte = sum([abs(e) for e in self.cte_history]) / len(self.cte_history)
        print(f"\n--- M6 Performance Metrics ---")
        print(f"Max Cross-Track Error: {max_cte:.3f} m")
        print(f"Avg Cross-Track Error: {avg_cte:.3f} m")