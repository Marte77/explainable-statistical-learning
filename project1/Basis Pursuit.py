import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import stft, istft
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

# --- CVXPY for convex optimization (Basis Pursuit) ---
import cvxpy as cp

# -----------------------------
# 1. Load audio & perform STFT
# -----------------------------
audio_path = "./1-137-A-32.wav"
y, sr = librosa.load(audio_path, sr=None)

# Optional: normalize the audio to avoid excessively large or small values
y = y / np.max(np.abs(y))

# STFT parameters
nperseg = 512
f, t, Zxx = stft(y, fs=sr, nperseg=nperseg)

# Separate the real and imaginary parts
Zxx_real = np.real(Zxx)  # shape = (freq_bins, time_frames)
Zxx_imag = np.imag(Zxx)  # used later to re-inject phase

freq_bins, time_frames = Zxx_real.shape
print(f"STFT shape (freq_bins, time_frames): {freq_bins}, {time_frames}")

# -----------------------------
# 2. Construct a random dictionary (you can replace this with a learned one)
# -----------------------------
n_atoms = 512  # number of dictionary atoms
# Dimensional match: each column is one atom => dictionary shape = (freq_bins, n_atoms)
D = np.random.randn(freq_bins, n_atoms).astype(np.float32)

# -----------------------------
# 3. Basis Pursuit solver function (CVXPY)
# -----------------------------
def solve_basis_pursuit(D, z, exact=True, eps=1e-3):
    """
    Solve Basis Pursuit with CVXPY:
      minimize ||x||_1 
      subject to D@x = z   (exact=True) 
         or 
         ||D@x - z||_2 <= eps  (exact=False)
    D: shape (freq_bins, n_atoms)
    z: shape (freq_bins,) the target vector to fit
    exact: whether to use strict equality constraint
    eps: if exact=False, the tolerance for residual
    returns: sparse solution x as a numpy vector shape (n_atoms,)
    """
    n_atoms = D.shape[1]
    
    # Define the optimization variable
    x = cp.Variable(n_atoms)
    
    # Define the constraint
    if exact:
        constraints = [D @ x == z]
    else:
        constraints = [cp.norm2(D @ x - z) <= eps]
    
    # Define the objective function
    objective = cp.Minimize(cp.norm1(x))
    
    # Build and solve the problem
    prob = cp.Problem(objective, constraints)
    # Choose a solver suitable for L1 problems, e.g., 'ECOS' or 'SCS'
    # If the exact constraint is not feasible, prob.solve() may fail or raise an error.
    result = prob.solve(solver=cp.SCS, verbose=False)
    
    if x.value is None:
        # Indicates an infeasible problem or no solution found
        return None
    return x.value

# -----------------------------
# 4. Solve BP for each frame to get sparse coefficients
# -----------------------------
sparse_coeffs = np.zeros((n_atoms, time_frames), dtype=np.float32)

# Decide whether to use exact equality (exact=True) or allow some noise (exact=False)
use_exact = False  # you can switch this to True to see if it's feasible

eps = 1e-2  # if use_exact=False, this controls the allowed residual

for i in range(time_frames):
    z_i = Zxx_real[:, i].astype(np.float32)
    x_sol = solve_basis_pursuit(D, z_i, exact=use_exact, eps=eps)
    
    if x_sol is None:
        # If not feasible, do some fallback treatment, e.g., set coefficients to zero
        print(f"Frame {i} is infeasible under exact constraint.")
        x_sol = np.zeros(n_atoms, dtype=np.float32)
    
    sparse_coeffs[:, i] = x_sol

# -----------------------------
# 5. Reconstruct the real part, then re-inject the imaginary part for iSTFT
# -----------------------------
Zxx_reconstructed_real = D @ sparse_coeffs  # shape = (freq_bins, time_frames)
Zxx_reconstructed = Zxx_reconstructed_real + 1j * Zxx_imag

_, y_reconstructed = istft(Zxx_reconstructed, fs=sr, nperseg=nperseg)

# -----------------------------
# 6. Compute metrics (SNR, MSE, PCC)
# -----------------------------
def compute_metrics(original, reconstructed):
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]
    noise = original - reconstructed
    
    snr = 10 * np.log10(np.sum(original**2) / np.sum(noise**2) + 1e-12)
    mse = mean_squared_error(original, reconstructed)
    pcc, _ = pearsonr(original, reconstructed)
    return snr, mse, pcc

snr, mse, pcc = compute_metrics(y, y_reconstructed)

# -----------------------------
# 7. Display results and save audio
# -----------------------------
print(f"SNR: {snr:.2f} dB, MSE: {mse:.6f}, PCC: {pcc:.4f}")

plt.figure()
plt.plot(y, label='Original')
plt.plot(y_reconstructed, '--', label='Reconstructed (BP)')
plt.legend()
plt.title(f"BP Reconstruction\nSNR={snr:.2f} dB, MSE={mse:.6f}, PCC={pcc:.4f}")
plt.show()

sf.write('reconstructed_audio_bp.wav', y_reconstructed, sr)
print("Reconstructed audio saved as 'reconstructed_audio_bp.wav'")
