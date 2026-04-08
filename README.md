# APT-Grid
APeriodic Turbomachinery Grid Generator
by Adekola Adeyemi, Justin Smart, Tony Woo, and Jeff Defoe
2024-2026

08 April 2026

Makes OH structured meshes for turbomachinery blade passage CFD simulations.
Has capability to generate conformal meshes for individual passages of aperiodic blade rows where each blade shape is different.
Key requirement for such blade rows is that the meriodional projection of all blades must be the same.

Currently, to run:
1. in the Python folder, modify the input block of bladePassageSurfaceGenerator_v2.py (in main, search for "INPUTS:")
2. in the Python folder, run bladePassageSurfaceGenerator_v2.py
3. in the top-level folder, make runtest.sh executable (chmod u+x runtest.sh)
4. in the top-level folder, run ./runtest.sh

Before running the above steps, put the hub, casing, and blade curve files in a folder called "inputData" and create an empty folder as follows: "mkdir -p constant/geometry"
See "README_fileFormatting.txt" in the inputData folder for requirements for the curve files.

The runtest script copies over files, updates blockMeshDict, generates the mesh, checks quality, and checks ability to form periodics.

Open issues:

1. Cutting at 2%/0% to clean up offset curve fraction maps is arbitrary, would be better to automate this. This seems to be a balancing act between non-orthogonal faces and cell size variation near the LE/TE.
2. Currently fails on cases where the extreme blade sections are fully within the annulus, i.e. they don't touch the hub or casing.
3. No testing yet conducted on actual aperiodic blade rows.
