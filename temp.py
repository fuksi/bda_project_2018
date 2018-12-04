import numpy as np
import time

raw_data = np.loadtxt('kristel.txt')

mat = np.zeros(shape=(23, 6))
values = np.arange(46,0,-1)

for v in values:
    for c in range(0, 6):
        for r in range(0, 23):
            if (c+r) == (46-v):
                mat[r][c] = v
                
tournament_result = []
for i in raw_data:
    r = int(i[0] + 12 - 1)
    c = int(6 - i[1])
    tournament_result.append(mat[r][c])
    
print(tournament_result)
print(min(tournament_result))
print(max(tournament_result))
filename = f'data/tournament_{time.time()}'
with open(filename, 'w') as f:
    for item in tournament_result:
        f.write(f'{int(item)}\n')