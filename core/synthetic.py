import numpy as np
import pandas as pd

class SyntheticDataGenerator:
    """Generates synthetic cosmological datasets for testing pipeline recovery and bias."""

    def __init__(self, seed=42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.c = 299792.458

    def generate_universe(self, 
                          h0_true=73.0, 
                          kappa_true=0.0, 
                          d_cmb_true=0.0, 
                          m_b_true=-19.25,
                          n_calibrators=42, 
                          n_hubble_flow=300):
        """
        Generates a synthetic universe with known parameters.
        """
        # --- 1. Calibrators ---
        cal_z = self.rng.uniform(0.001, 0.01, size=n_calibrators)
        # Log-normal distribution of X_i typical of galaxies
        cal_xi = self.rng.lognormal(mean=np.log(1e-7), sigma=0.5, size=n_calibrators)
        
        # Directions for CMB dipole test
        # Generate random directions on sphere
        u = self.rng.uniform(-1, 1, size=n_calibrators)
        theta = self.rng.uniform(0, 2*np.pi, size=n_calibrators)
        cal_x = np.sqrt(1 - u**2) * np.cos(theta)
        cal_y = np.sqrt(1 - u**2) * np.sin(theta)
        cal_z_dir = u
        
        # True distance modulus
        # DL = c * z / H0 (approximation for very low z)
        cal_dl = self.c * cal_z / h0_true
        cal_mu_true = 5 * np.log10(cal_dl) + 25
        
        # Inject TEP signals (Clock - Candle)
        # TRGB is non-clock (unaffected by TEP)
        # Cepheid is clock (affected by kappa and dipole)
        # We assume dipole is along z_dir for simplicity in synthetic tests
        cal_mu_trgb = cal_mu_true + self.rng.normal(0, 0.05, size=n_calibrators)
        
        tep_signal = kappa_true * cal_xi + d_cmb_true * cal_z_dir
        cal_mu_cep = cal_mu_true + tep_signal + self.rng.normal(0, 0.05, size=n_calibrators)
        
        df_calibrators = pd.DataFrame({
            "galaxy": [f"SYNTH_{i}" for i in range(n_calibrators)],
            "z": cal_z,
            "X_i": cal_xi,
            "dir_x": cal_x,
            "dir_y": cal_y,
            "dir_z": cal_z_dir,
            "mu_true": cal_mu_true,
            "mu_TRGB": cal_mu_trgb,
            "mu_Cep": cal_mu_cep,
            "mu_TRGB_err": 0.05,
            "mu_Cep_err": 0.05
        })
        
        # --- 2. Hubble Flow SNe ---
        # Log-uniform redshift distribution between 0.023 and 0.15
        hf_z = np.exp(self.rng.uniform(np.log(0.023), np.log(0.15), size=n_hubble_flow))
        hf_xi = self.rng.lognormal(mean=np.log(1e-7), sigma=0.5, size=n_hubble_flow)
        
        # True DL
        hf_dl = self.c * hf_z / h0_true
        hf_mu_true = 5 * np.log10(hf_dl) + 25
        
        # SN magnitudes (affected by Cepheid zero-point calibration in a real pipeline)
        hf_m_b = hf_mu_true + m_b_true + self.rng.normal(0, 0.1, size=n_hubble_flow)
        
        df_hubble_flow = pd.DataFrame({
            "sn_id": [f"SN_SYNTH_{i}" for i in range(n_hubble_flow)],
            "zcmb": hf_z,
            "X_i": hf_xi,
            "mu_true": hf_mu_true,
            "m_b_corr": hf_m_b,
            "m_b_corr_err_DIAG": 0.1
        })
        
        return df_calibrators, df_hubble_flow

