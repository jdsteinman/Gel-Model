"""Parameter studies in Providas and Kattis
'Finite element method in plane Cosserat elasticity'

Note: We are using N rather than a as the coupling
parameter

Author: Jack S. Hale 2014 mail@jackhale.co.uk
"""
import subprocess
import itertools

import pandas as pd
import dolfin as df

from plate_with_hole import plate_with_hole


def main():
    tables_four_and_five()


def tables_four_and_five():
    ls = [0.216/1.063, 0.216/10.63]
    Ns = [0, 0.25, 0.5, 0.75, 0.9]

    mesh = df.Mesh('meshes/plate_with_circular_hole/very_fine.xml')

    results = []
    for l, N in itertools.product(ls, Ns):
        result = plate_with_hole(mesh, l, N)
        result['l'] = l
        result['N'] = N
        result['r'] = 0.216
        result['R'] = 16.2
        result['r/l'] = result['r']/result['l']
        results.append(result)

    frame = pd.DataFrame(results)
    results = frame.sort_values(by=['r/l', 'N'])
    print results
    try:
        results.to_clipboard()
    except RuntimeError:
        print "Normally results are copied to the clipboard, but no windowing system is running."
    else:
        print "The results above have been copied into the clipboard"
        print "You can paste them directly into e.g. Excel"

def cmd_exists(cmd):
    return subprocess.call("type " + cmd, shell=True,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE) == 0


if __name__ == "__main__":
    main()
