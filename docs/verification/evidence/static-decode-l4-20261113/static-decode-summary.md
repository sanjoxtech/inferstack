| profile | prompt words | batch | native now ms/step | static eager | static graph | speedup (graph vs now) | greedy agreement | max logit diff |
|---|---|---|---|---|---|---|---|---|
| native_unfused | 48 | 1 | 43.3 | 43.4 | 18.4 | 2.35x | 1.000 | 0.188 |
| native_unfused | 48 | 4 | 46.1 | 44.6 | 21.5 | 2.14x | 1.000 | 0.312 |
| native_unfused | 48 | 16 | 46.3 | 45.2 | 29.1 | 1.59x | 1.000 | 0.531 |
| native_unfused | 400 | 1 | 43.2 | 43.2 | 18.8 | 2.30x | 1.000 | 0.156 |
| native_unfused | 400 | 4 | 46.4 | 44.7 | 22.5 | 2.06x | 0.996 | 0.188 |
| native_unfused | 400 | 16 | 46.1 | 45.0 | 33.5 | 1.38x | 0.996 | 0.281 |
| native_full | 48 | 1 | error: TypeError: 'NoneType' object is not callable |
| native_full | 48 | 4 | error: TypeError: 'NoneType' object is not callable |
| native_full | 48 | 16 | error: TypeError: 'NoneType' object is not callable |
| native_full | 400 | 1 | error: TypeError: 'NoneType' object is not callable |
| native_full | 400 | 4 | error: TypeError: 'NoneType' object is not callable |
| native_full | 400 | 16 | error: TypeError: 'NoneType' object is not callable |
