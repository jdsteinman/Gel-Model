#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Parameter studies in Nakamura and Lakes
'Stress concentration around a blunt crack'

Author: Jack S. Hale 2014 mail@jackhale.co.uk
"""

import os
import subprocess
import itertools
from string import Template
import tempfile

import numpy as np
import pandas as pd
import dolfin as df
from dolfin_utils.meshconvert import meshconvert

from plate_with_elliptical_hole import plate_with_ellipse


def main():
    if not cmd_exists('gmsh'):
        raise RuntimeError('You must have gmsh installed to run this script.')

    table_one()


def table_one():
    # parameters from Nakamura and Lakes
    # length scale
    ls = [0.1, 1.0]
    # coupling
    Ns = [0.0, 0.93]
    # major axis length over radius of curvature
    gs = [1.00, 2.50, 5.00]
    # radius of curvative
    r = 0.5

    # iterate over product space of parameters
    results = []
    for g, N, l in itertools.product(gs, Ns, ls):
        major_axis_length = g*r
        minor_axis_length = major_axis_length*np.sqrt(1.0/g)

        # Comment: Changing y_dim=9.0 and running with
        # symmetry boundary condition is worth exploring.
        mesh = generate_elliptical_mesh(y_dim=9.0, x_dim=12.7,
                                        major_axis_length=major_axis_length,
                                        minor_axis_length=minor_axis_length)

        result = plate_with_ellipse(mesh, l, N)
        result['c/r'] = g
        result['N'] = N
        result['l'] = l
        result['x_dim'] = 25.4
        result['y_dim'] = 18.0
        result['major_axis_length (c)'] = major_axis_length
        result['minor_axis_length (b)'] = minor_axis_length
        result['radius_of_curvature (r)'] = r
        results.append(result)

    frame = pd.DataFrame(results)
    result = frame.sort_values(by=['c/r', 'l', 'N'])
    print result
    try:
        result.to_clipboard()
    except RuntimeError:
        print "Normally results are copied to the clipboard, but no windowing system is running."
    else:
        print "The results above have been copied into the clipboard"
        print "You can paste them directly into e.g. Excel"


def generate_elliptical_mesh(major_axis_length,
                             minor_axis_length,
                             x_dim, y_dim):
    # read gmsh geometry file into string
    with open("meshes/plate_with_elliptical_hole/parameterised.geo",
              "r") as geometry_in_file:
        geometry_in = geometry_in_file.read()
        geometry_in = Template(geometry_in)
        # substitute in our parameters for this run
        geometry_out = geometry_in.substitute(x_dim=x_dim,
                                              y_dim=y_dim,
                                              major_axis_length=major_axis_length,
                                              minor_axis_length=minor_axis_length)
        geometry_in_file.close()

    # can't seem to get gmsh to work with pipes etc. so write to tmp file
    # open temporary file
    with tempfile.NamedTemporaryFile(mode='w') as geometry_out_file:
        geometry_out_file.write(geometry_out)
        geometry_out_file.flush()

        # generate mesh with gmsh
        subprocess.call(['gmsh', '-2',
                        geometry_out_file.name])
        msh_out_filename = geometry_out_file.name + '.msh'

        # convert mesh to dolfin format
        xml_out_filename = geometry_out_file.name + '.xml'
        meshconvert.convert2xml(msh_out_filename, xml_out_filename,
                                iformat='gmsh')

        mesh = df.Mesh(xml_out_filename)

        # cleanup
        os.remove(msh_out_filename)
        os.remove(xml_out_filename)
        geometry_out_file.close()

    return mesh


def cmd_exists(cmd):
    return subprocess.call("type " + cmd, shell=True,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE) == 0


if __name__ == "__main__":
    main()
