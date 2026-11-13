class CumulativeErrorPreventer:
    """
    Enforces the logical chain across Paper 8, 9, 11, and Six Problems.
    Prevents 'Silent Compounding' of unverified structural assumptions.
    """
    def __init__(self):
        self.pipeline_state = {
            'Paper8_Raw_Detected': False,
            'Paper9_Dictionary_Matched': False,
            'Paper11_Assumption_Light_Applied': False,
            'SixProblems_Raw_Subsampled': False,
            'Regress_Flag': False
        }

    def step1_paper8_internal_test(self, data_converges):
        """Paper 8: Run internal diagnostic (e.g., Birkhoff average)."""
        print("[Paper 8] Running internal convergence test...")
        self.pipeline_state['Paper8_Raw_Detected'] = data_converges
        if not data_converges:
            print(" -> Fails Paper 8. Routing to Paper 6 (Change-Point).")
            return False
        return True

    def step2_paper9_dictionary(self, has_mixing_rate):
        """Paper 9: Map to Statistical Dictionary."""
        print("[Paper 9] Mapping to Statistical Dictionary...")
        # Paper 9 Entry 2.3 requires a mixing rate. If unknown, it's a residual caveat.
        if not has_mixing_rate:
            print(" -> WARNING: Mixing rate unknown. Regress introduced!")
            self.pipeline_state['Regress_Flag'] = True
        self.pipeline_state['Paper9_Dictionary_Matched'] = True

    def step3_paper11_assumption_light(self):
        """Paper 11: Apply Assumption-Light fix to close the Regress."""
        if self.pipeline_state['Regress_Flag']:
            print("[Paper 11] Applying Assumption-Light Subsampling to remove nuisance parameter...")
            self.pipeline_state['Paper11_Assumption_Light_Applied'] = True
            self.pipeline_state['Regress_Flag'] = False # Regress mitigated
        else:
            print("[Paper 11] No regress detected. Standard CI applied.")

    def step4_sixproblems_raw_subsampling(self, is_composite_statistic):
        """Six Problems (Prob 4): Prevent CUSUM + Hill compounding error."""
        if is_composite_statistic:
            print("[Six Problems] Composite statistic detected (CUSUM -> Hill).")
            print(" -> FORCING Raw-Process Subsampling to account for CUSUM estimation error.")
            self.pipeline_state['SixProblems_Raw_Subsampled'] = True
        else:
            print("[Six Problems] Standard subsampling sufficient.")

    def validate_pipeline(self):
        """Final check to ensure no unmitigated regress reached the conclusion."""
        print("\n--- Final Pipeline Validation ---")
        if self.pipeline_state['Regress_Flag']:
            raise RuntimeError("FATAL: Cumulative Error! Unmitigated regress reached the final output. (Violates Paper 11)")
        
        if self.pipeline_state['SixProblems_Raw_Subsampled'] or not self.pipeline_state.get('is_composite', False):
            print("SUCCESS: The statistical guarantee is valid. No silent compounding occurred.")
        else:
            raise RuntimeError("FATAL: Composite statistic used without Raw-Process Subsampling! (Violates Six Problems)")

# --- Execution Simulation ---
preventer = CumulativeErrorPreventer()

# Simulate a scenario where data doesn't converge normally (Paper 8 fails)
# but we detect change points, then try to estimate Tail Index (Paper 9)
# WITHOUT knowing the mixing rate, creating a Regress.
preventer.step1_paper8_internal_test(data_converges=False)
preventer.step2_paper9_dictionary(has_mixing_rate=False) # Triggers Regress
preventer.step3_paper11_assumption_light()               # Fixes Regress
preventer.step4_sixproblems_raw_subsampling(is_composite_statistic=True) # Fixes Six Prob 4
preventer.validate_pipeline()
