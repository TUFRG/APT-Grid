#!/bin/bash

# Jeff Defoe, May 2026
# Reads single blade file per folder that NAX produces and moves them to
# individually-named files all in the same folder.

set -e

bladePrefix=IGVBlade
cd inputData
for dir in */; do
	namenoslash=${dir%/}
	echo "Blade folder: ${namenoslash}"
	num=$(echo ${namenoslash##*[!0-9]})
	echo "Number = $num"
	echo "Copying curve file and renaming..."
    cp ${dir}*.curve ${bladePrefix}.curve$num
done
