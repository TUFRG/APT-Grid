# bladePassageGridGenerator
Structured grid generator for turbomachinery blade passages

03 December 2025

Everything has now been combined to a single script, scriptRev14.py.

Once fixed, simply run:

python scriptRev14.py

and then:

./runtest.sh in the folder of the OpenFOAM case setup, e.g. ../../hGridSep2025/template/
		This copies over files, updates blockMeshDict, generates the mesh, checks quality, and checks ability to form periodics.

However, I haven't fixed a few things to make it operate seamlessly yet:

Open issues:

1. The offset arclength fraction mapping still writes out files which later get read back in, but the reading and writing locations are different (previously requiring copying files). Need to fix so these variables just are pulled from memory, removing the file-writing altogether.

2. Currently some of the arclength mapping functions had initial offset (2nd column) values where they didn't start at exactly 0.0; need to enforce this.

3. Manually have been cutting data in the txt files at 2% for the sections not already getting cut at 4%. Need to implement this automatically and also determine how to know if 2% is sensisble. This seems to be a balancing act between non-orthogonal faces and cell size variation near the LE/TE.

4. No testing has been conducted on other geometries. Need to test on Justin's ECL5 stator and IGV.

5. Current code doesn't have full integration of multiple passage meshing for aperiodic blade rows. Main thing to sort out here is that the code is piecemeal; once all functionality in Python is integrated into a single code, it is _hoped_ that the way everything has been set up will ensure that aperiodic passages are conformal geometrically and for all mesh points on shared patches.
