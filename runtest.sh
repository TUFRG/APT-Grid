#!/bin/bash

set -e

clear
cp ~/Documents/Work/UWindsor/Research/OpenFOAMturbomachinery/structuredMeshWithProjection/TUFRG/surfaceGenerator/ECL5_original/periodic/passageParameters system/
cp ~/Documents/Work/UWindsor/Research/OpenFOAMturbomachinery/structuredMeshWithProjection/TUFRG/surfaceGenerator/ECL5_original/periodic../constant/geometry1/*.stl constant/geometry/
./geomUpdate.sh
blockMesh
checkMesh
foamToVTK -faceSet nonOrthoFaces
createPatch
rm -rf 1
