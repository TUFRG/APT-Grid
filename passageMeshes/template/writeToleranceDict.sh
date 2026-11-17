#!/bin/bash

sed -i "/^pointMergeTol/ s/pointMergeTol .*/pointMergeTol ${AG_POINT_MERGE_TOL:-0.7};/" constant/toleranceDict
sed -i "/^edgeMergeTol/ s/edgeMergeTol .*/edgeMergeTol ${AG_EDGE_MERGE_TOL:-0.00006};/" constant/toleranceDict
sed -i "/^integralAdjTol/ s/integralAdjTol .*/integralAdjTol ${AG_INTEGRAL_ADJ_TOL:-0.9};/" constant/toleranceDict
