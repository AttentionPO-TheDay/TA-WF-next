# Burst RLE round-trip audit

The encoder emits one signed integer per maximal equal-direction run; sign is direction and absolute value is exact run length. The decoder repeats each sign by its absolute length. This is reversible run-length coding of packet direction, not filtering or information removal.

| date | exact checks | failures |
|---|---:|---:|
| day0 | 18,553 | 0 |
| day14 | 22,603 | 0 |
| day30 | 22,867 | 0 |
| day90 | 28,599 | 0 |
| day150 | 24,064 | 0 |
| day270 | 19,935 | 0 |

All 136,621 canonical traces passed exact element-wise decode(encode(x)) == x, total run length == valid packet length, nonzero run length, and alternating adjacent signs. Interpretation was therefore allowed to continue. Maximal-run direction alternation is mechanical and was not treated as an independent feature.
