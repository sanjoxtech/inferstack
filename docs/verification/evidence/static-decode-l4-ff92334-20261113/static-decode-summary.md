| profile | prompt words | batch | native now ms/step | static eager | graph + torch GDN | graph + Triton GDN | speedup (graph vs now) | greedy agreement | max logit diff |
|---|---|---|---|---|---|---|---|---|---|
| native_unfused | 48 | 1 | 45.0 | 45.1 | 18.4 | 17.6 | 2.56x | 1.000 | 0.156 |
| native_unfused | 48 | 4 | 47.7 | 45.4 | 21.5 | 19.6 | 2.44x | 1.000 | 0.281 |
| native_unfused | 48 | 16 | 48.1 | 46.3 | 29.0 | 23.5 | 2.05x | 1.000 | 0.531 |
| native_unfused | 400 | 1 | 44.3 | 44.0 | 18.7 | 18.0 | 2.46x | 1.000 | 0.156 |
| native_unfused | 400 | 4 | 48.3 | 45.1 | 22.5 | 20.4 | 2.37x | 0.996 | 0.188 |
| native_unfused | 400 | 16 | 48.0 | 45.7 | 33.4 | 27.8 | 1.72x | 0.996 | 0.379 |
| native_full | 48 | 1 | 30.0 | 38.2 | 17.2 | 16.4 | 1.83x | 1.000 | 0.156 |
| native_full | 48 | 4 | 35.4 | 40.1 | 20.3 | 18.3 | 1.94x | 1.000 | 0.375 |
| native_full | 48 | 16 | 36.4 | 40.0 | 27.9 | 22.2 | 1.64x | 0.999 | 0.500 |
| native_full | 400 | 1 | 29.9 | 38.4 | 17.6 | 16.7 | 1.78x | 0.984 | 0.156 |
| native_full | 400 | 4 | 35.4 | 39.6 | 21.4 | 19.1 | 1.85x | 0.996 | 0.219 |
| native_full | 400 | 16 | 35.3 | 40.1 | 32.1 | 26.4 | 1.34x | 0.994 | 0.312 |
