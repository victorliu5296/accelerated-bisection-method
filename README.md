# accelerated-bisection-method
Experiments on applying the Aitken delta-squared acceleration method for faster convergence of sequences in combination with the bisection method for root-finding. There is also a hybrid method that combines the bisection and the secant method, it seems to be the fastest method in practice (the tested functions are well-behaved).

To try it out, you can copy-paste the `hybrid_bisection.cpp` file into an online C++ compiler and run it. The results are printed to the console.
There is also a short python script that was for early prototyping.