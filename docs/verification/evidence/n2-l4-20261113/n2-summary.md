| profile | batch | chunk | prefix cache | wall ms | model ms | snapshot ms | other ms | forwards | served TTFT p50 (L4) |
|---|---|---|---|---|---|---|---|---|---|
| native_unfused | 1 | 128 | True | 978 [974-987] | 960 | 13 | 3 | 13 | 1135.6 |
| native_unfused | 1 | 128 | False | 967 [967-969] | 964 | 0 | 3 | 13 |  |
| native_unfused | 1 | 512 | True | 401 [399-401] | 396 | 3 | 2 | 4 |  |
| native_unfused | 1 | 512 | False | 403 [402-405] | 402 | 0 | 2 | 4 |  |
| native_unfused | 1 | 2048 | True | 214 [213-217] | 213 | 0 | 1 | 1 |  |
| native_unfused | 1 | 2048 | False | 218 [217-220] | 216 | 0 | 1 | 1 |  |
| native_unfused | 4 | 128 | True | 1005 [997-1007] | 1001 | 0 | 3 | 13 | 1009.8 |
| native_unfused | 4 | 128 | False | 1001 [997-1005] | 998 | 0 | 3 | 13 |  |
| native_unfused | 4 | 512 | True | 854 [853-855] | 851 | 0 | 2 | 4 |  |
| native_unfused | 4 | 512 | False | 857 [855-858] | 854 | 0 | 2 | 4 |  |
| native_unfused | 4 | 2048 | True | 963 [960-964] | 961 | 0 | 2 | 1 |  |
| native_unfused | 4 | 2048 | False | 965 [963-967] | 963 | 0 | 2 | 1 |  |
| native_full | 1 | 128 | True | 868 [868-874] | 853 | 12 | 3 | 13 | 1016.6 |
| native_full | 1 | 128 | False | 854 [850-860] | 851 | 0 | 3 | 13 |  |
| native_full | 1 | 512 | True | 372 [359-376] | 367 | 3 | 2 | 4 |  |
| native_full | 1 | 512 | False | 361 [359-362] | 359 | 0 | 1 | 4 |  |
| native_full | 1 | 2048 | True | 201 [200-203] | 199 | 0 | 1 | 1 |  |
| native_full | 1 | 2048 | False | 202 [200-203] | 201 | 0 | 1 | 1 |  |
| native_full | 4 | 128 | True | 889 [882-890] | 886 | 0 | 3 | 13 | 898.5 |
| native_full | 4 | 128 | False | 888 [888-890] | 884 | 0 | 3 | 13 |  |
| native_full | 4 | 512 | True | 807 [805-810] | 804 | 0 | 3 | 4 |  |
| native_full | 4 | 512 | False | 803 [801-815] | 801 | 0 | 3 | 4 |  |
| native_full | 4 | 2048 | True | 841 [840-841] | 839 | 0 | 2 | 1 |  |
| native_full | 4 | 2048 | False | 840 [839-841] | 838 | 0 | 2 | 1 |  |
