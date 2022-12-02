
import subprocess
import json
fils = [
    'cola/logs/18128965.out',
    'mnli/logs/18198602.out',
    'mrpc/logs/18222610.out',
    'qqp/logs/18172083.out',
    'rte/logs/18224321.out',
    'snli/logs/18174243.out',
    'sst2/logs/18251000.out',
    'qnli/logs/18256128.out'
]

d = {}
o = []
for e in fils:
    name = e.split ('/')[0]
    o.append (name)
    v = json.loads (subprocess.check_output (f"tail -n 3 {e}|sed -n '1p'", shell = True).decode ())
    for key, value in v.items ():
        if key not in d.keys ():
            d[ key ] = {}
        d[key][ name ] = value

print (' '.join (['Easyness'] + o))
for key, value in d.items ():
    s = str(key)        
    for e in o:
        if e in value.keys ():
            s += ' ' + str (value[e])
        else:
            s += ' 0'
    print (s)
