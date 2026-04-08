#!/bin/bash

set -e

clear
rm -rf VTK
mv outputData/passage0/*.stl constant/geometry/
mv outputData/passage0/passageParameters system/
./geomUpdate.sh
blockMesh
checkMesh
createPatch
rm -rf 1
foamToVTK -faceSet nonOrthoFaces
