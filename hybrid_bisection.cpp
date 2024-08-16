#include <iostream>
#include <functional>
#include <chrono>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <deque>

const int MAX_ITERATIONS = 1000;
const double EPSILON = 1e-10;
const int BISECTION_THRESHOLD = 3;

struct BenchmarkResult
{
    double root;
    int iterations;
    int function_evaluations;
    double elapsed_time_us;
};

double aitken_accelerate(double x0, double x1, double x2)
{
    return x2 - ((x2 - x1) * (x2 - x1)) / (x2 - 2 * x1 + x0);
}

static BenchmarkResult aitken_bisection(const std::function<double(double)> &f, double a, double b, double tol = 1e-6, int max_iter = 100)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    int function_evaluations = 2; // Initial evaluations for f(a) and f(b)
    int iterations = 0;

    double fa = f(a), fb = f(b);
    if (fa * fb >= 0)
    {
        throw std::runtime_error("Function values at interval endpoints must have opposite signs");
    }

    while ((b - a) > tol && iterations < max_iter)
    {
        // Perform three bisection steps
        double c1 = (a + b) / 2;
        double fc1 = f(c1);
        function_evaluations++;

        if (fc1 * fa < 0)
        {
            b = c1;
            fb = fc1;
        }
        else
        {
            a = c1;
            fa = fc1;
        }

        double c2 = (a + b) / 2;
        double fc2 = f(c2);
        function_evaluations++;

        if (fc2 * fa < 0)
        {
            b = c2;
            fb = fc2;
        }
        else
        {
            a = c2;
            fa = fc2;
        }

        double c3 = (a + b) / 2;
        double fc3 = f(c3);
        function_evaluations++;

        // Apply Aitken's acceleration
        double denominator = c3 - 2 * c2 + c1;
        if (std::abs(denominator) > 1e-10) // Avoid division by very small numbers
        {
            double c_aitken = c1 - std::pow(c2 - c1, 2) / denominator;

            // Check if c_aitken satisfies IVT
            if (a < c_aitken && c_aitken < b)
            {
                double fc_aitken = f(c_aitken);
                function_evaluations++;

                if (fc_aitken * fa < 0)
                {
                    b = c_aitken;
                    fb = fc_aitken;
                }
                else if (fc_aitken * fb < 0)
                {
                    a = c_aitken;
                    fa = fc_aitken;
                }
                else
                {
                    if (fc3 * fa < 0)
                    {
                        b = c3;
                        fb = fc3;
                    }
                    else
                    {
                        a = c3;
                        fa = fc3;
                    }
                }
            }
            else
            {
                if (fc3 * fa < 0)
                {
                    b = c3;
                    fb = fc3;
                }
                else
                {
                    a = c3;
                    fa = fc3;
                }
            }
        }
        else
        {
            if (fc3 * fa < 0)
            {
                b = c3;
                fb = fc3;
            }
            else
            {
                a = c3;
                fa = fc3;
            }
        }

        iterations++;
    }

    double root = (a + b) / 2;
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    return {root, iterations, function_evaluations, duration.count() / 1000.0};
}

static BenchmarkResult improved_hybrid_bisection(const std::function<double(double)> &f, double a, double b)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    int function_evaluations = 2; // Initial evaluations for f(a) and f(b)

    if (f(a) * f(b) >= 0)
    {
        throw std::runtime_error("Function values at interval endpoints must have opposite signs");
    }

    double fa = f(a), fb = f(b);
    double x[3] = {a, b, 0};
    int x_index = 2;
    double fx[3] = {fa, fb, 0};
    int consecutive_improvements = 0;
    double last_improvement = std::abs(b - a);

    for (int i = 0; i < MAX_ITERATIONS; ++i)
    {
        double c;
        if (i < 2 || consecutive_improvements < 2)
        {
            c = (a + b) / 2; // Bisection
        }
        else
        {
            // Secant method
            c = b - fb * (b - a) / (fb - fa);
            if (c <= a || c >= b)
            { // Fallback to bisection if out of bounds
                c = (a + b) / 2;
            }
        }

        x[x_index] = c;
        double fc = f(c);
        fx[x_index] = fc;
        function_evaluations++;

        if (std::abs(fc) < EPSILON || std::abs(b - a) < EPSILON)
        {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
            return {c, i + 1, function_evaluations, duration.count() / 1000.0};
        }

        if (fa * fc < 0)
        {
            b = c;
            fb = fc;
        }
        else
        {
            a = c;
            fa = fc;
        }

        double improvement = std::abs(b - a);
        if (improvement < last_improvement)
        {
            consecutive_improvements++;
            if (consecutive_improvements >= 3)
            {
                // Apply Aitken acceleration
                double accelerated = aitken_accelerate(x[0], x[1], x[2]);
                if (a <= accelerated && accelerated <= b)
                {
                    double f_acc = f(accelerated);
                    function_evaluations++;
                    if (std::abs(f_acc) < std::abs(fc))
                    {
                        c = accelerated;
                        fc = f_acc;
                        if (fa * fc < 0)
                        {
                            b = c;
                            fb = fc;
                        }
                        else
                        {
                            a = c;
                            fa = fc;
                        }
                    }
                }
            }
        }
        else
        {
            consecutive_improvements = 0;
        }
        last_improvement = improvement;

        x_index = (x_index + 1) % 3;
    }

    throw std::runtime_error("Improved hybrid method failed to converge within the maximum number of iterations");
}

static BenchmarkResult regular_bisection(const std::function<double(double)> &f, double a, double b)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    int function_evaluations = 2; // Initial evaluations for f(a) and f(b)

    if (f(a) * f(b) >= 0)
    {
        throw std::runtime_error("Function values at interval endpoints must have opposite signs");
    }

    double fa = f(a), fb = f(b);

    for (int i = 0; i < MAX_ITERATIONS; ++i)
    {
        double c = (a + b) / 2;

        if (std::abs(b - a) < 2 * EPSILON)
        {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
            return {c, i + 1, function_evaluations, duration.count() / 1000.0};
        }

        double fc = f(c);
        function_evaluations++;

        if (fa * fc < 0)
        {
            b = c;
            fb = fc;
        }
        else
        {
            a = c;
            fa = fc;
        }
    }

    throw std::runtime_error("Regular bisection method failed to converge within the maximum number of iterations");
}

class Benchmark
{
public:
    struct Result
    {
        double elapsed_time_us;
        int iterations;
        int function_evaluations;
        double root;
    };

    static Result run_hybrid(const std::function<double(double)> &f, double a, double b)
    {
        auto result = improved_hybrid_bisection(f, a, b);
        return {result.elapsed_time_us, result.iterations, result.function_evaluations, result.root};
    }

    static Result run_regular(const std::function<double(double)> &f, double a, double b)
    {
        auto result = regular_bisection(f, a, b);
        return {result.elapsed_time_us, result.iterations, result.function_evaluations, result.root};
    }

    static Result run_aitken(const std::function<double(double)> &f, double a, double b)
    {
        auto result = aitken_bisection(f, a, b);
        return {result.elapsed_time_us, result.iterations, result.function_evaluations, result.root};
    }

    static void benchmark(const std::function<double(double)> &f, double a, double b, int num_runs)
    {
        std::vector<Result> hybrid_results, regular_results, itp_results, aitken_results;

        try
        {
            for (int i = 0; i < num_runs; ++i)
            {
                hybrid_results.push_back(run_hybrid(f, a, b));
                regular_results.push_back(run_regular(f, a, b));
                aitken_results.push_back(run_aitken(f, a, b));
            }
        }
        catch (const std::exception &e)
        {
            std::cout << "Error: " << e.what() << "\n";
        }

        auto calculate_stats = [](const std::vector<Result> &results)
            -> std::tuple<double, double, double, double, double, double, double>
        {
            if (results.empty())
            {
                return std::make_tuple(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            }
            std::vector<double> times;
            double sum_iterations = 0, sum_evaluations = 0;
            for (const auto &result : results)
            {
                times.push_back(result.elapsed_time_us);
                sum_iterations += result.iterations;
                sum_evaluations += result.function_evaluations;
            }
            double sum = std::accumulate(times.begin(), times.end(), 0.0);
            double mean = sum / times.size();
            double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
            double stdev = std::sqrt(sq_sum / times.size() - mean * mean);
            auto min_max = std::minmax_element(times.begin(), times.end());
            double avg_iterations = sum_iterations / results.size();
            double avg_evaluations = sum_evaluations / results.size();
            return std::make_tuple(mean, stdev, *min_max.first, *min_max.second, avg_iterations, avg_evaluations, results[0].root);
        };

        auto print_stats = [](const char *method, const std::tuple<double, double, double, double, double, double, double> &stats)
        {
            std::cout << method << " approach:\n"
                      << "  Mean time: " << std::fixed << std::setprecision(2) << std::get<0>(stats) << " microseconds\n"
                      << "  StdDev time: " << std::fixed << std::setprecision(2) << std::get<1>(stats) << " microseconds\n"
                      << "  Min time: " << std::fixed << std::setprecision(2) << std::get<2>(stats) << " microseconds\n"
                      << "  Max time: " << std::fixed << std::setprecision(2) << std::get<3>(stats) << " microseconds\n"
                      << "  Avg iterations: " << std::fixed << std::setprecision(2) << std::get<4>(stats) << "\n"
                      << "  Avg function evaluations: " << std::fixed << std::setprecision(2) << std::get<5>(stats) << "\n"
                      << "  Root approximation: " << std::setprecision(15) << std::get<6>(stats) << "\n";
        };

        print_stats("Improved Hybrid Bisection", calculate_stats(hybrid_results));
        print_stats("Regular Bisection", calculate_stats(regular_results));
        print_stats("ITP Method", calculate_stats(itp_results));
        print_stats("Aitken-Bisection Method", calculate_stats(aitken_results));
    }
};

int main()
{
    std::vector<std::pair<std::function<double(double)>, std::pair<double, double>>> test_functions = {
        {[](double x)
         { return x * x - 2; },
         {0, 2}}, // f(x) = x^2 - 2, root at sqrt(2)
        {[](double x)
         { return std::cos(x) - x; },
         {0, 1}}, // f(x) = cos(x) - x, root near 0.739085
        {[](double x)
         { return std::exp(x) - 3 * x; },
         {0, 1}}, // f(x) = e^x - 3x, root near 0.61906
        {[](double x)
         { return std::sin(x) / x - 0.5; },
         {0.5, 2}}, // f(x) = sin(x)/x - 0.5, root near 1.89549
        {[](double x)
         { return std::log(x) + std::sqrt(x) - 5; },
         {1, 20}}, // f(x) = ln(x) + sqrt(x) - 5, root near 8.30943
        {[](double x)
         { return std::tan(x) + x; },
         {2, 3}}, // f(x) = tan(x) + x, root near 2.02876
    };

    int num_runs = 1000;

    for (const auto &test_case : test_functions)
    {
        std::cout << "Testing function " << &test_case - &test_functions[0] + 1 << ":\n";
        try
        {
            Benchmark::benchmark(test_case.first, test_case.second.first, test_case.second.second, num_runs);
        }
        catch (const std::exception &e)
        {
            std::cout << "  Error: " << e.what() << "\n";
        }
        std::cout << "\n";
    }

    return 0;
}