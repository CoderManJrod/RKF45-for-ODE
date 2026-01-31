#Programmers: Jared Walker
#Packages used: math, time, argparse, numpy, and matplotlib
#Implementation approach:
#Implemented a single-step RKF45 (Fehlberg) method using k1-k6.
#Computes both 4th-order and 5th-order estimates each step.
#Uses |y5 - y4| as a local truncation error estimate.
#Implements Euler’s method as a traditional baseline method.
#Runs 1,000 steps (1,001 points) for the original case and two variations.
#Prints computational steps performed and actual computing time.
#Plots RKF45 vs Euler and (optionally) a difference plot for visibility.




import math
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    #ODE: y' = -y + ln(x)
    return -y + math.log(x)


def rkf45_step(x, y, h):
    #Fehlberg RKF45 coefficients
    k1 = f(x, y)

    k2 = f(x + h/4.0,
           y + h*(k1/4.0))

    k3 = f(x + 3.0*h/8.0,
           y + h*(3.0*k1/32.0 + 9.0*k2/32.0))

    k4 = f(x + 12.0*h/13.0,
           y + h*(1932.0*k1/2197.0 - 7200.0*k2/2197.0 + 7296.0*k3/2197.0))

    k5 = f(x + h,
           y + h*(439.0*k1/216.0 - 8.0*k2 + 3680.0*k3/513.0 - 845.0*k4/4104.0))

    k6 = f(x + h/2.0,
           y + h*(-8.0*k1/27.0 + 2.0*k2 - 3544.0*k3/2565.0 + 1859.0*k4/4104.0 - 11.0*k5/40.0))

    #4th-order estimate
    y4 = y + h*(25.0*k1/216.0 + 1408.0*k3/2565.0 + 2197.0*k4/4104.0 - 1.0*k5/5.0)

    #5th-order estimate
    y5 = y + h*(16.0*k1/135.0 + 6656.0*k3/12825.0 + 28561.0*k4/56430.0 - 9.0*k5/50.0 + 2.0*k6/55.0)

    err_est = abs(y5 - y4)
    return x + h, y5, y4, err_est


def euler_step(x, y, h):
    return x + h, y + h*f(x, y)


def run_case(label, x0, y0, h, n_steps):
    x_rkf = np.zeros(n_steps + 1, dtype=float)
    y_rkf = np.zeros(n_steps + 1, dtype=float)
    x_eul = np.zeros(n_steps + 1, dtype=float)
    y_eul = np.zeros(n_steps + 1, dtype=float)
    err_est = np.zeros(n_steps, dtype=float)

    x_rkf[0], y_rkf[0] = x0, y0
    x_eul[0], y_eul[0] = x0, y0

    #Timing RKF45
    t0 = time.perf_counter()
    for i in range(n_steps):
        xn, yn = x_rkf[i], y_rkf[i]
        xn1, y5, y4, e = rkf45_step(xn, yn, h)
        x_rkf[i + 1] = xn1
        y_rkf[i + 1] = y5
        err_est[i] = e
    t1 = time.perf_counter()
    rkf_time = t1 - t0

    #Timing Euler
    t2 = time.perf_counter()
    for i in range(n_steps):
        xn, yn = x_eul[i], y_eul[i]
        xn1, yn1 = euler_step(xn, yn, h)
        x_eul[i + 1] = xn1
        y_eul[i + 1] = yn1
    t3 = time.perf_counter()
    eul_time = t3 - t2

    #Compare RKF vs Euler
    diff = np.abs(y_rkf - y_eul)
    max_diff = float(np.max(diff))
    final_diff = float(diff[-1])

    #RKF error estimate summary
    max_est = float(np.max(err_est))
    mean_est = float(np.mean(err_est))
    final_est = float(err_est[-1])

    summary = {
        "label": label,
        "x0": x0, "y0": y0, "h": h, "n_steps": n_steps,
        "rkf_time": rkf_time,
        "eul_time": eul_time,
        "max_diff_rkf_vs_euler": max_diff,
        "final_diff_rkf_vs_euler": final_diff,
        "max_rkf_err_est": max_est,
        "mean_rkf_err_est": mean_est,
        "final_rkf_err_est": final_est,
        "x_rkf": x_rkf, "y_rkf": y_rkf,
        "x_eul": x_eul, "y_eul": y_eul
    }

    return summary


def print_summary(s):
    print("\n" + "="*60)
    print(f"CASE: {s['label']}")
    print(f"ODE: y' = -y + ln(x)")
    print(f"Start: x0={s['x0']}, y0={s['y0']}, h={s['h']}, steps={s['n_steps']}  ->  points={s['n_steps']+1}")
    print("-"*60)
    print(f"RKF45: steps={s['n_steps']}, runtime={s['rkf_time']:.6f} sec")
    print(f"Euler: steps={s['n_steps']}, runtime={s['eul_time']:.6f} sec")
    print("-"*60)
    print("RKF45 local error estimate |y5 - y4|:")
    print(f"  max={s['max_rkf_err_est']:.6e}, mean={s['mean_rkf_err_est']:.6e}, last={s['final_rkf_err_est']:.6e}")
    print("Difference between methods |y_RKF - y_Euler|:")
    print(f"  max={s['max_diff_rkf_vs_euler']:.6e}, final={s['final_diff_rkf_vs_euler']:.6e}")
    print("="*60)


def plot_case(s, show=True, save_prefix=None):
    plt.figure()
    plt.plot(s["x_rkf"], s["y_rkf"], label="RKF45 (5th order)")
    plt.plot(s["x_eul"], s["y_eul"], label="Euler")
    plt.title(f"ODE Solution Comparison ({s['label']})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(2, 30)
    plt.ylim(0.8, 3.2)
    plt.legend()
    plt.grid(True)

    #Extra plot: absolute difference between methods to make differences visible
    plt.figure()
    diff = np.abs(s["y_rkf"] - s["y_eul"])
    plt.plot(s["x_rkf"], diff, label="|RKF45 - Euler|")
    plt.title(f"Method Difference ({s['label']})")
    plt.xlabel("x")
    plt.ylabel("Absolute difference")
    plt.legend()
    plt.grid(True)

    if save_prefix:
        plt.savefig(f"{save_prefix}_{s['label'].replace(' ', '_').lower()}_diff.png", dpi=200)

    if show:
        plt.show()


    if save_prefix:
        plt.savefig(f"{save_prefix}_{s['label'].replace(' ', '_').lower()}.png", dpi=200)

    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000, help="number of steps (points = steps + 1)")
    parser.add_argument("--h", type=float, default=0.3, help="step size")
    parser.add_argument("--x0", type=float, default=2.0, help="initial x")
    parser.add_argument("--y0", type=float, default=1.0, help="initial y")
    parser.add_argument("--saveplots", action="store_true", help="save plots as PNG files")
    args = parser.parse_args()

    # Make sure x stays positive (ln(x) domain)
    if args.x0 <= 0:
        raise ValueError("x0 must be > 0 because ln(x) is used.")
    if args.x0 + args.steps*args.h <= 0:
        raise ValueError("x must remain > 0 over the interval.")

    #Original case
    case1 = run_case("Original", args.x0, args.y0, args.h, args.steps)

    #Two variations (simple + valid)
    case2 = run_case("Variation A (different y0)", args.x0, args.y0 + 0.5, args.h, args.steps)
    case3 = run_case("Variation B (smaller h)", args.x0, args.y0, args.h/2.0, args.steps)

    for s in (case1, case2, case3):
        print_summary(s)

    save_prefix = "project2" if args.saveplots else None
    plot_case(case1, show=True, save_prefix=save_prefix)
    plot_case(case2, show=True, save_prefix=save_prefix)
    plot_case(case3, show=True, save_prefix=save_prefix)


if __name__ == "__main__":
    main()
