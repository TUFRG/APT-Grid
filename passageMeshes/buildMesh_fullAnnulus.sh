#!/bin/bash

# by Adekola Adeyemi, Oct. 2026
# generalized by Jeff Defoe

set -e

passages=$(ls ../outputData/ | wc -l)
passagem1=$((passages - 1))
passagem2=$((passages - 2))

echo 'Creating meshes for each passage...'
for i in $(seq 0 "$passagem1"); do
	echo "Creating mesh for passage$i ..."
	mkdir -p passage$i;
	cp -r ../template/* passage$i/;
	cd passage$i;
	cp ../../outputData/passage$i/*.stl constant/geometry/;
	cp ../../outputData/passage$i/passageParameters system/;
	./geomUpdate.sh;
	sed -i "s/pCyclic/pCyclic$i/" system/blockMeshDict;
	sed -i "s/nCyclic/nCyclic$i/" system/blockMeshDict;
	blockMesh;
	checkMesh;
	cd ..;
done 
echo 'All passage meshes created!'
echo ' '
echo 'Merging and stitching passage meshes together...'
for j in $(seq 0 "$passagem2"); do
	echo "Merging passage$j to passage0 ...";
	mergeMeshes passage0 passage$((j+1)) -overwrite;
	cd passage0
	rm -rf 0
	echo 'Stitching overlapping patches...'
	stitchMesh -overwrite pCyclic$j nCyclic$((j+1));
	rm -rf 0
	cd ..
done
cd passage0
stitchMesh -overwrite pCyclic$passagem1 nCyclic0
rm -rf 0

