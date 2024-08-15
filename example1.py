import math

def f(x):
    return x**3 - x - 2

def bisection(f, a, b, tol=1e-6, max_iter=100):
    iterations = 0
    while (b - a) > tol and iterations < max_iter:
        c = (a + b) / 2
        if f(c) == 0:
            return c, iterations
        elif f(c) * f(a) < 0:
            b = c
        else:
            a = c
        iterations += 1
    return (a + b) / 2, iterations

def aitkin_bisection(f, a, b, tol=1e-6, max_iter=100):
    iterations = 0
    while (b - a) > tol and iterations < max_iter:
        # Perform three bisection steps
        c1 = (a + b) / 2
        if f(c1) * f(a) < 0:
            b = c1
        else:
            a = c1
        
        c2 = (a + b) / 2
        if f(c2) * f(a) < 0:
            b = c2
        else:
            a = c2
        
        c3 = (a + b) / 2
        
        # Apply Aitkin's acceleration
        denominator = c3 - 2*c2 + c1
        if abs(denominator) > 1e-10:  # Avoid division by very small numbers
            c_aitkin = c1 - (c2 - c1)**2 / denominator
            
            # Check if c_aitkin satisfies IVT
            if a < c_aitkin < b:
                if f(c_aitkin) * f(a) < 0:
                    b = c_aitkin
                elif f(c_aitkin) * f(b) < 0:
                    a = c_aitkin
                else:
                    if f(c3) * f(a) < 0:
                        b = c3
                    else:
                        a = c3
            else:
                if f(c3) * f(a) < 0:
                    b = c3
                else:
                    a = c3
        else:
            if f(c3) * f(a) < 0:
                b = c3
            else:
                a = c3
        
        iterations += 1
    
    return (a + b) / 2, iterations

# Run both methods
a, b = 1, 2
root_bisection, iter_bisection = bisection(f, a, b, 1e-12)
root_aitkin, iter_aitkin = aitkin_bisection(f, a, b, 1e-12)

print(f"Standard Bisection: Root = {root_bisection:.15f}, Iterations = {iter_bisection}")
print(f"Aitkin Bisection: Root = {root_aitkin:.15f}, Iterations = {iter_aitkin}")
print(f"Actual Root: x≈1.521379706804567569604081")

# Output:
# ```
# Standard Bisection: Root = 1.521379706804964, Iterations = 40
# Aitkin Bisection: Root = 1.521379706804562, Iterations = 15
# Actual Root: x≈1.521379706804567569604081
# ```
