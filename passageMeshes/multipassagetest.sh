#!/bin/bash

set -e

clear

mkdir -p passage0
cp -r template/* passage0/
cd passage0
cp ../../outputData/passage0/*.stl constant/geometry/
cp ../../outputData/passage0/passageParameters system/
./geomUpdate.sh
sed -i 's/pCyclic/pCyclic0/' system/blockMeshDict
blockMesh
checkMesh
cd ..

mkdir -p passage1
cp -r template/* passage1/
cd passage1
cp ../../outputData/passage1/*.stl constant/geometry/
cp ../../outputData/passage1/passageParameters system/
./geomUpdate.sh
sed -i 's/nCyclic/nCyclic1/' system/blockMeshDict
blockMesh
checkMesh
cd ..

mergeMeshes passage0 passage1
cd passage0
stitchMesh -perfect pCyclic0 nCyclic1
