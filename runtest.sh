#!/bin/bash

set -e

clear
rm -rf VTK
cp ~/Documents/Work/UWindsor/Research/OpenFOAMturbomachinery/structuredMeshWithProjection/TUFRG/surfaceGenerator/ECL5_original/periodic/passageParameters system/
cp ~/Documents/Work/UWindsor/Research/OpenFOAMturbomachinery/structuredMeshWithProjection/TUFRG/surfaceGenerator/ECL5_original/periodic../constant/geometry1/*.stl constant/geometry/
./geomUpdate.sh
blockMesh
checkMesh
createPatch
rm -rf 1
foamToVTK -faceSet nonOrthoFaces
