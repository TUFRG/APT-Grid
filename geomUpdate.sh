#!/bin/bash

while read line; do

	echo 'line value:'
	echo $line
	echo ' '

	lineToken=$(echo $line | cut -f1 -d ' ')
	lineValue=$(echo $line | cut -f2 -d ' ')

	echo 'token:'
	echo $lineToken
	echo 'value:'
	echo $lineValue
	echo ' '

	sed -i "/^${lineToken}/ s/${lineToken} .*/${lineToken} ${lineValue}/" system/blockMeshDict

done <system/passageGeometry
