# EZDSP: Easy Digital Signal Processing

## Key Features
- **PyTorch Integration**: With functions written in PyTorch ([`./src/ezdsp/nn/`](./src/ezdsp/nn/)), EZDSP leverages the power of parallel computation, making it ideal for machine learning applications.
- **Automatic Data Handling**: Thanks to the [torch_fn decorator](https://github.com/ywatanabe1989/torch_fn), EZDSP functions accept torch.tensors (CPU & GPU), numpy.ndarray, and pd.DataFrame as input and return in the corresponding data type.
- **Practical Examples**: [`./example.py`](./example.py) demonstrates all of the functions listed below and generates output figures in [`./example_outputs/`](./example_outputs/).

## Installation
```bash
$ pip install ezdsp
```

## Quick Start
``` python
import ezdsp

# Parameters
SRC_FS = 1024  # Source sampling frequency
TGT_FS = 512   # Target sampling frequency
FREQS_HZ = [10, 30, 100]  # Frequencies in Hz
LOW_HZ = 20    # Low frequency for bandpass filter
HIGH_HZ = 50   # High frequency for bandpass filter
SIGMA = 10     # Sigma for Gaussian filter


# Demo Signal
xx, tt, fs = ezdsp.demo_sig(
    t_sec=T_SEC, fs=SRC_FS, freqs_hz=FREQS_HZ, sig_type="chirp"
)
# xx is compatible with torch.tensor (on cpu / cuda), numpy.ndarray, or pd.DataFrame.

# Normalization
xx_norm = ezdsp.norm.z(xx)
xx_minmax = ezdsp.norm.minmax(xx)

# Resampling
xx_resampled = ezdsp.resample(xx, fs, TGT_FS)

# Noise addition
xx_gauss = ezdsp.add_noise.gauss(xx)
xx_white = ezdsp.add_noise.white(xx)
xx_pink = ezdsp.add_noise.pink(xx)
xx_brown = ezdsp.add_noise.brown(xx)

# Filtering
xx_filted_bandpass = ezdsp.filt.bandpass(xx, fs, low_hz=LOW_HZ, high_hz=HIGH_HZ)
xx_filted_bandstop = ezdsp.filt.bandstop(xx, fs, low_hz=LOW_HZ, high_hz=HIGH_HZ)
xx_filted_gauss = ezdsp.filt.gauss(xx, sigma=SIGMA)

# Hilbert Transformation
phase, amplitude = ezdsp.hilbert(xx) # or envelope

# Wavelet Transformation
wavelet_coef, wavelet_freqs = ezdsp.wavelet(xx, fs)

# Power Spetrum Density
psd, psd_freqs = ezdsp.psd(xx, fs)

# Phase-Amplitude Coupling
pac, freqs_pha, freqs_amp = ezdsp.pac(x_3d, fs) # This function is computationally demanding. Please monitor the RAM/VRAM usage.
```

# Citation
To cite EZDSP in your work, please use the following format:
``` bibtex
@misc{ezdsp2024,
  author = {Watanabe, Yusuke},
  title = {{EZDSP: Easy Digital Signal Processing}},
  year = {2023},
  howpublished = {\url{https://github.com/ywatanabe1989/ezdsp}},
}
```

# Contact
For further inquiries or contributions, please contact Yusuke WATANABE (ywata1989@gmail.com).
