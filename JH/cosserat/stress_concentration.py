#!/usr/bin/python
# -*- coding: utf-8 -*-

from itertools import product

import dolfin as df
import pandas as pd

from plate_with_hole import plate_with_hole
import analytical

def main():
    mesh = df.Mesh("meshes/plate_with_circular_hole/crazy_fine.xml")
    ls = [0.216/1.063, 0.216/10.63]
    Ns = [0.001, 0.25, 0.5, 0.75, 0.9]
    nu = 0.3
    R = 0.216

    results = []
    for l, N in product(ls, Ns):
        result = plate_with_hole(mesh, l, N)
        result['analytical_stress_concentration'] = analytical.plate_with_hole.stress_concentration(nu, R, l, N)
        results.append(result)

    frame = pd.DataFrame(results)
    print frame

if __name__ == "__main__":
    main()
