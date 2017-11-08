# 2048

## Requirements
* [numba](http://numba.pydata.org/)

## Performance
* Using Numba and totally random play, initial results (total time for 10 000 games):
  * games: 10000, time: 0.626307, avg_score: 1051.317200, avg_move_count: 116.330300
* Simple MC search (for each move run n random plays, pick move with best total score):
  * larger n helps, but not that much. even n=100 gets to 2048 sometimes. n=1000 gets you there somewhat more often
  * Seems move_count is better evaluator than score?
