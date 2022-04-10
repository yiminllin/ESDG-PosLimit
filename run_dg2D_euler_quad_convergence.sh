#!/bin/bash
for N in 1 2 3 4
do
for K1D in 2 4 8 16 32
do
  nohup julia -t 4 examples/IDP/dg2D_euler_quad_convergence.jl $N $K1D >> dg2D_euler_quad_convergence_hist.log 2>&1 &
  wait $!
done
done
