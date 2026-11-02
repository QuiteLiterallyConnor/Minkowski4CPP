Dataset: ModelNet10. Architectures and input features may differ between methods.

| method | seeds | accuracy | macro P | macro R | macro F1 | train s | latency b1 ms | throughput /s | peak VRAM train+eval MB | peak RAM MB | active/sample | convert ms | params |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform | 1,2,3 | 74.5 ± 2.8 | 76.1 ± 2.4 | 74.5 ± 2.8 | 74.3 ± 2.9 | 12.0 ± 2.8 | 37.36 ± 5.13 | 446 ± 40 | 57 ± 0 | 3777 ± 24 | 786 ± 0 | 0.22 ± 0.04 | 677930 ± 0 |
| adaptive | 1,2,3 | 78.0 ± 1.7 | 80.2 ± 1.2 | 78.0 ± 1.7 | 77.6 ± 1.7 | 6.7 ± 0.4 | 10.86 ± 0.36 | 948 ± 11 | 171 ± 0 | 4078 ± 13 | 369 ± 0 | 0.98 ± 0.23 | 882650 ± 0 |
| gaussian | 1,2,3 | 79.5 ± 2.6 | 82.3 ± 1.2 | 79.5 ± 2.6 | 78.4 ± 3.5 | 2.6 ± 0.2 | 2.31 ± 0.12 | 6738 ± 225 | 42 ± 0 | 3771 ± 0 | 147 ± 0 | 6.07 ± 0.17 | 110218 ± 0 |
| adaptive_gaussian | 1,2,3 | 80.8 ± 1.3 | 82.2 ± 1.5 | 80.8 ± 1.3 | 80.6 ± 1.0 | 5.8 ± 0.3 | 11.85 ± 1.66 | 1135 ± 74 | 94 ± 0 | 3949 ± 1 | 190 ± 0 | 5.93 ± 0.64 | 885242 ± 0 |

Test accuracy (%) per epoch, mean over matched seeds:

| epoch | uniform | adaptive | gaussian | adaptive_gaussian |
|---|---|---|---|---|
| 1 | 15.7 | 63.8 | 13.2 | 63.5 |
| 2 | 67.5 | 69.3 | 65.0 | 75.0 |
| 3 | 76.3 | 72.8 | 71.3 | 80.3 |
| 4 | 75.5 | 75.2 | 76.7 | 80.2 |
| 5 | 76.0 | 75.7 | 78.7 | 80.5 |
| 6 | 74.5 | 78.0 | 79.5 | 80.8 |
