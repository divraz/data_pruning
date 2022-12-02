import sys
start = int (sys.argv[1])
end = int (sys.argv[2])

import subprocess
z = [0.75, 0.5, 0.4, 0.3, 0.25, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005]
k = {}
for i in range (len (z)):
    k[ str (z[i]) ] = i

d = {'random':[0]*len(k), '1':[0]*len(k), '2':[0]*len(k), '3':[0]*len(k)}

for e in range (start, end + 1):
    x = subprocess.check_output (f"sed -n '1p;57p' {e}.out", shell = True).decode ()
    x = x.split ('\n')
    a = x[0].split ('_')[-3]
    b = x[0].split ('_')[-4]
    d[ b ][ k[a] ] = round (float (x[1].split ()[-3].replace (',', '')) * 100, 2)

for key, values in d.items ():
    print (key)
    for value in values:
        print (value)
    print ()
