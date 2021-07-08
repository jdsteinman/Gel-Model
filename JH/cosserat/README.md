# Cosserat elasticity finite element solver

This repository contains a finite element solver for the planar Cosserat
elasticity problem using the DOLFIN finite element problem solving environment
from the [FEniCS Project](http://www.fenicsproject.org).

If you find this solver helpful for your research, we kindly ask that you
consider citing the paper:

Nano-structured materials: inhomogeneities and imperfect interfaces in plane micropolar elasticity, a boundary element approach.
Elena Atroshchenko, Jack S. Hale, Javier A. Videla, Stanislav Potapenko and Stéphane P.A. Bordas
Pre-print submitted.

This repository and the Docker image required to run it is permanently archived
on figshare with the DOI:

https://dx.doi.org/10.6084/m9.figshare.4047462

## Author

Jack S. Hale (University of Luxembourg)

# Instructions

Clone this repository:

    git clone git@bitbucket.org:unilucompmech/cosserat.git

Install Docker (or FEniCS and gmsh) following the instructions at [docker.com](https://docker.com).       

Change directory into the folder containing the source:

    cd cosserat

Launch a Docker container:

    ./docker-run

Convert the gmsh geometry files to DOLFIN XML files:

    ./convert_meshes.py

Run an analysis, e.g.:

    python plate_with_inclusion.py

## Description of files

`elastic_plate_with_inclusion.py`: Normal elasticity, plate with inclusion.
`nakamura_lakes.py`: Study from Nakamura and Lakes.
`patch_tests.py`: Patch tests.
`plate_with_elliptical_hole.py`: Cosserat elliptical hole solver.
`plate_with_hole.py`: Cosserat circular hole solver.
`plate_with_inclusion.py`: Cosserat circular inclusion solver.
`providas_kattis.py`: Study from Providas and Kattis.
`stress_concentration.py`: Another circular hole study. 
`weak_form.py`: Variational forms for all studies, used Voigt notation.
`analytical/elastic_plate_with_inclusion.py`: Stress concentrations elastic plate circular inclusion.
`analytical/plate_with_hole.py`: Stress concentration Cosserat plate with circular inclusion.

## License

cosserat is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

cosserat is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with cosserat. If not, see <http://www.gnu.org/licenses/>.