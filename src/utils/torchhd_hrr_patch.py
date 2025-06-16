# Monkey-patched to preserve identity on single-slot inputs:
# The default HRRTensor.multibind/bundle implementations always run their FFT- or sum-based
# operations even when there is only one vector to bind or bundle, which both semantically and
# numerically should act as the identity but instead distorts the result. By short-circuiting
# when the “property” dimension is 1, we bypass any self-binding/bundling, return the input
# unchanged, and avoid precision drift from circular convolution or majority-vote aggregation.
import torch
from torchhd import HRRTensor

_orig_multibind = HRRTensor.multibind
_orig_multibundle = HRRTensor.multibundle  # inherited from VSATensor if not overridden


def hrr_multibind_corrected(self) -> "HRRTensor":
    """
    Original implementation of multibind():
    --- torchhd.tensors.hrr.HRRTensor.multibind
    result = ifft(torch.prod(fft(self), dim=-2, dtype=self.dtype))
    return torch.real(result)
    ---
    The culprit lies in the torch.prod(..., dtype=self.dtype) part of the implementation. Here's a breakdown:
    - fft(self) produces a complex tensor (the Fourier transforms of the input vectors). For real input vectors of
        dtype float32, fft yields a complex64 result (with real and imaginary parts).
    - The code then calls torch.prod(..., dtype=self.dtype). Here, self.dtype is a real type (e.g. torch.float32).
        This forces the product to be computed in a float dtype, essentially discarding the imaginary component of the
        complex numbers during multiplication. In other words, the frequency-domain data is being cast back to real
        prematurely.
    - As a result, torch.prod does not correctly multiply the complex spectra. It likely multiplies only the real parts
        (or otherwise drops the imaginary parts to fit into torch.float32), which is not the correct operation for
        convolution. By the time ifft() is called, the frequency-domain information is incomplete/corrupted
        (no imaginary part), so the inverse FFT yields the wrong vector.

    In summary, the implementation bug is that HRRTensor.multibind performs the multiplication in the wrong dtype.
    The imaginary components that are crucial for correct circular convolution are lost. The sequential binding (a.bind(b))
    does not have this issue because it multiplies the complex spectra directly (using torch.mul on the FFT results without
    changing dtype) and only takes the real part at the end.

    With this implementation:
    a.bind(b) == torchhd.multibind(torch.stack([a, b], dim=0))
    as expected

    """
    # Compute FFT of all vectors (complex result)
    spectra = torch.fft.fft(self, dim=-1)
    # Multiply spectra across vectors (no dtype cast to float!)
    prod_spectra = torch.prod(spectra, dim=-2)
    # Inverse FFT to get convolution result
    result = torch.fft.ifft(prod_spectra, dim=-1)
    # Then take the real part
    return torch.real(result).as_subclass(HRRTensor)


def patched_multibind(self):
    if self.size(-2) == 1:
        # identity on single‐slot
        return self.squeeze(-2)
    return hrr_multibind_corrected(self)


def patched_multibundle(self):
    if self.size(-2) == 1:
        # identity on single‐slot
        return self.squeeze(-2)
    return _orig_multibundle(self)


# apply the patch
HRRTensor.multibind = patched_multibind
HRRTensor.multibundle = patched_multibundle

