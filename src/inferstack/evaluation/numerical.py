"""Numerical equivalence oracle measured in ULPs of the output dtype.

Fixed absolute tolerances are wrong for low-precision outputs: one BF16 unit in
the last place is 0.0625 at magnitude 8-16, so a correctly rounded kernel fails
`max_abs <= 3e-2`. Instead, every candidate output is compared with an FP32 (or
FP64) oracle computed from the same low-precision inputs, and the error is
expressed in ULPs of the candidate's output dtype at the oracle's magnitude.

Pass criterion (per tensor):
- max error <= `max_ulp` (1.0 for single-rounded elementwise/normalization ops;
  larger, declared bounds for reductions/recurrences), and
- mean error <= `max_mean_ulp`;
- OR, when a reference implementation is supplied, the candidate is no worse
  than that reference (max and mean) — the reference itself may double-round.
Outputs computed in FP32 (e.g. recurrent state) use a relative bound instead.
Inputs are array-likes; torch tensors are converted via `.float().cpu().numpy()`.
"""
from __future__ import annotations

import math

import numpy as np

MANTISSA_BITS = {'bfloat16': 7, 'float16': 10, 'float32': 23}
MIN_NORMAL_EXP = {'bfloat16': -126, 'float16': -14, 'float32': -126}


def as_array(x):
    if hasattr(x, 'detach'):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def ulp_error(value, oracle, dtype):
    """Elementwise |value - oracle| in ULPs of `dtype` at the oracle's magnitude."""
    if dtype not in MANTISSA_BITS:
        raise ValueError(f'unsupported dtype {dtype}')
    v, o = as_array(value), as_array(oracle)
    if v.shape != o.shape:
        raise ValueError(f'shape mismatch {v.shape} vs {o.shape}')
    if not np.all(np.isfinite(v)):
        return np.full(v.shape, np.inf)
    exponent = np.floor(np.log2(np.maximum(np.abs(o), 2.0 ** MIN_NORMAL_EXP[dtype])))
    return np.abs(v - o) / np.exp2(exponent - MANTISSA_BITS[dtype])


def check(name, value, oracle, *, dtype, max_ulp=1.0, max_mean_ulp=0.5, reference=None):
    oracle_finite = bool(np.all(np.isfinite(as_array(oracle))))
    errors = ulp_error(value, oracle, dtype)
    result = {'check': name, 'dtype': dtype, 'max_ulp': float(errors.max()), 'mean_ulp': float(errors.mean()),
              'frac_over_1ulp': float((errors > 1.0 + 1e-9).mean()), 'bound_max_ulp': max_ulp,
              'bound_mean_ulp': max_mean_ulp, 'elements': int(errors.size)}
    if errors.size == 0 or not oracle_finite or not np.all(np.isfinite(errors)):
        # Non-finite candidate/oracle values or an empty comparison can never pass,
        # including through the no-worse-than-reference branch (inf <= inf).
        result.update({'basis': 'nonfinite' if errors.size else 'empty', 'pass': False})
        return result
    within = math.isfinite(result['max_ulp']) and result['max_ulp'] <= max_ulp and result['mean_ulp'] <= max_mean_ulp
    if reference is not None:
        ref = ulp_error(reference, oracle, dtype)
        if not np.all(np.isfinite(ref)):
            result.update({'reference_max_ulp': None, 'reference_mean_ulp': None})
            result['basis'] = 'ulp_bound' if within else 'failed'
            result['pass'] = bool(within)
            return result
        result.update({'reference_max_ulp': float(ref.max()), 'reference_mean_ulp': float(ref.mean())})
        no_worse = result['max_ulp'] <= result['reference_max_ulp'] + 1e-9 and \
            result['mean_ulp'] <= result['reference_mean_ulp'] + 1e-9
        result['basis'] = 'ulp_bound' if within else 'no_worse_than_reference' if no_worse else 'failed'
        within = within or no_worse
    else:
        result['basis'] = 'ulp_bound' if within else 'failed'
    result['pass'] = bool(within)
    return result


def check_relative(name, value, oracle, *, max_rel=1e-4, floor=1e-6):
    """For FP32-computed outputs (e.g. recurrent state): max relative error with an absolute floor."""
    v, o = as_array(value), as_array(oracle)
    if v.size == 0 or not (np.all(np.isfinite(v)) and np.all(np.isfinite(o))):
        return {'check': name, 'dtype': 'float32', 'max_rel': None, 'mean_rel': None, 'bound_max_rel': max_rel,
                'elements': int(v.size), 'pass': False, 'basis': 'nonfinite' if v.size else 'empty'}
    rel = np.abs(v - o) / np.maximum(np.abs(o), floor)
    result = {'check': name, 'dtype': 'float32', 'max_rel': float(rel.max()), 'mean_rel': float(rel.mean()),
              'bound_max_rel': max_rel, 'elements': int(rel.size)}
    result['pass'] = bool(np.all(np.isfinite(v)) and result['max_rel'] <= max_rel)
    result['basis'] = 'relative_bound' if result['pass'] else 'failed'
    return result


def round_to(x, dtype):
    """Round-to-nearest-even emulation of bfloat16/float16 in numpy (for tests and CPU oracles)."""
    a = np.asarray(x, dtype=np.float32)
    if dtype == 'float16':
        return a.astype(np.float16).astype(np.float64)
    if dtype != 'bfloat16':
        return a.astype(np.float64)
    bits = a.view(np.uint32).astype(np.uint64)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16
    return rounded.astype(np.uint32).view(np.float32).astype(np.float64)
