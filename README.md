# bladePassageGridGenerator
Structured grid generator for turbomachinery blade passages

03 December 2025

Everything has now been combined to a single script, scriptRev14.py.

Once fixed, simply run:
python scriptRev14.py
or:
ipython
then:
run ./scriptRev14.py

and then:

./runtest.sh in the folder of the OpenFOAM case setup, e.g. ../../hGridSep2025/template/
		This copies over files, updates blockMeshDict, generates the mesh, checks quality, and checks ability to form periodics.

Open issues:

1. Cutting at 4%/2% to clean up offset curve fraction maps is arbitrary, would be better to automate this. This seems to be a balancing act between non-orthogonal faces and cell size variation near the LE/TE.

2. No testing has been conducted on other geometries. Need to test on Justin's ECL5 stator and IGV.

3. Current code doesn't have full integration of multiple passage meshing for aperiodic blade rows. Main thing to sort out here is that the code is piecemeal; once all functionality in Python is integrated into a single code, it is _hoped_ that the way everything has been set up will ensure that aperiodic passages are conformal geometrically and for all mesh points on shared patches.
