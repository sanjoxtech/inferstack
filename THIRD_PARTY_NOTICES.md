# Third-party notices

InferStack is Apache-2.0. Parts below follow other open-source projects and keep their notices.

## flash-linear-attention (fla-core 0.6.0), MIT

`src/inferstack/kernels/fla_gdr_states.py` mirrors the forward of `fla/ops/gated_delta_rule/chunk.py`
(same internal ops, same order) so it can also return the per-chunk states. InferStack also calls
FLA kernels at runtime (optional dependency). Source: https://github.com/fla-org/flash-linear-attention

```
MIT License

Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Ideas only (no code copied)

- SGLang (Apache-2.0): radix/prefix caching, GDN prefill tracking, scheduler behaviour, studied as design references.
- vLLM (Apache-2.0): chunked prefill / goodput benchmarking concepts.
- SLO-Guard (arXiv 2604.17627), AutoTuneBench (arXiv 2609.18123), MLPerf Inference rules, DistServe (arXiv 2401.09670): SLO and measurement methodology.
