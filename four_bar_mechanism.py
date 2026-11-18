# Mohammad Zare
# Dynamics of Machinery (DOM) project
# Student ID: 40030570
#
# Kinematic analysis and simplified force analysis of a four-bar mechanism.
# Solves the position, velocity and acceleration loop-closure equations for
# the coupler and rocker links, then estimates the inertia forces on each
# link from the kinematics. Includes a Grashof check, solver convergence
# warnings, and an animation of the mechanism over one crank revolution.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import fsolve


def get_positive_input(prompt):
    """Prompt the user for a positive numeric input, re-prompting on invalid entries."""
    while True:
        try:
            value = float(input(prompt))
            if value <= 0:
                print("Please enter a value greater than zero.")
            else:
                return value
        except ValueError:
            print("Invalid input. Please enter a valid number.")


# Inputs
# Link lengths
l1 = get_positive_input('Please enter the length of the first link (Ground): ')
l2 = get_positive_input('Please enter the length of the second link: ')
l3 = get_positive_input('Please enter the length of the third link: ')
l4 = get_positive_input('Please enter the length of the fourth link: ')

# Angular velocity of the second link (crank)
omega2 = get_positive_input('Please enter the angular velocity of the second link: ')

# Masses of the links
m2 = get_positive_input('Please enter mass of the second link: ')
m3 = get_positive_input('Please enter mass of the third link: ')
m4 = get_positive_input('Please enter mass of the fourth link: ')

# Grashof check: the crank (link 2) can only make a full revolution if
# s + l <= p + q AND the crank is the shortest link.
lengths = sorted([l1, l2, l3, l4])
if not (lengths[0] + lengths[3] <= lengths[1] + lengths[2] and l2 == lengths[0]):
    print('Warning: with these lengths the crank cannot make a full revolution '
          '(Grashof condition not met). Some crank angles have no valid solution.')

# Range of crank angle theta2 over one full revolution
theta2_values = np.linspace(0, 2 * np.pi, 720)
alpha2 = 0  # Assume zero angular acceleration for the crank (constant omega2)


def equations_position(vars, theta2):
    """System of position, velocity and acceleration loop-closure equations
    for the four-bar linkage, to be solved simultaneously via fsolve."""
    theta3, theta4, omega3, omega4, alpha3, alpha4 = vars

    # Position (loop closure)
    eq1 = l3 * np.cos(theta3) - l4 * np.cos(theta4) - l1 + l2 * np.cos(theta2)
    eq2 = l3 * np.sin(theta3) - l4 * np.sin(theta4) + l2 * np.sin(theta2)

    # Velocity
    eq3 = l4 * omega4 * np.sin(theta4) - l3 * omega3 * np.sin(theta3) - l2 * omega2 * np.sin(theta2)
    eq4 = l2 * omega2 * np.cos(theta2) - l4 * omega4 * np.cos(theta4) + l3 * omega3 * np.cos(theta3)

    # Acceleration
    eq5 = (l2 * alpha2 * np.cos(theta2) - l2 * omega2 ** 2 * np.sin(theta2)
           + l3 * alpha3 * np.cos(theta3) - l3 * omega3 ** 2 * np.sin(theta3)
           - l4 * alpha4 * np.cos(theta4) + l4 * omega4 ** 2 * np.sin(theta4))
    eq6 = (-l2 * alpha2 * np.sin(theta2) - l2 * omega2 ** 2 * np.cos(theta2)
           - l3 * alpha3 * np.sin(theta3) - l3 * omega3 ** 2 * np.cos(theta3)
           + l4 * alpha4 * np.sin(theta4) + l4 * omega4 ** 2 * np.cos(theta4))

    return [eq1, eq2, eq3, eq4, eq5, eq6]


def equations_force(vars, alpha2, omega2):
    """Estimate the inertia forces on each link from its kinematic state (simplified model)."""
    theta2, theta3, theta4, omega3, omega4, alpha3, alpha4 = vars

    # Forces in X-direction for each link
    f2_x = m2 * (l2 * omega2 ** 2 * np.cos(theta2) + l2 * alpha2 * np.sin(theta2))
    f3_x = m3 * (l3 * omega3 ** 2 * np.cos(theta3) - l2 * omega2 ** 2 * np.cos(theta2))
    f4_x = m4 * (l4 * omega4 ** 2 * np.cos(theta4) - l3 * omega3 ** 2 * np.cos(theta3))

    # Forces in Y-direction for each link
    f2_y = m2 * (-l2 * omega2 ** 2 * np.sin(theta2) + l2 * alpha2 * np.cos(theta2))
    f3_y = m3 * (-l3 * omega3 ** 2 * np.sin(theta3) - l2 * omega2 ** 2 * np.sin(theta2))
    f4_y = m4 * (-l4 * omega4 ** 2 * np.sin(theta4) - l3 * omega3 ** 2 * np.sin(theta3))

    # Resultant force magnitude for each link
    f2 = np.sqrt(f2_x ** 2 + f2_y ** 2)
    f3 = np.sqrt(f3_x ** 2 + f3_y ** 2)
    f4 = np.sqrt(f4_x ** 2 + f4_y ** 2)

    return [f2_x, f2_y, f3_x, f3_y, f4_x, f4_y, f2, f3, f4]


# Arrays to store results
theta3_values = []
theta4_values = []
omega3_values = []
omega4_values = []
alpha3_values = []
alpha4_values = []
forces = []
failed_angles = []

# Initial guess for [theta3, theta4, omega3, omega4, alpha3, alpha4]
initial_guess = [np.pi / 4, np.pi / 2, 0, 0, 0, 0]  # theta3, theta4 must not start at 0: that guess is degenerate and fsolve gets stuck

# Solve the equations for each crank angle, using the previous solution
# as the initial guess for the next angle (continuation method)
for theta2 in theta2_values:
    solution, info, ier, msg = fsolve(equations_position, initial_guess, args=(theta2,), full_output=True)
    if ier != 1:  # fsolve did not converge: the mechanism cannot be assembled at this angle
        failed_angles.append(np.degrees(theta2))
    theta3_values.append(solution[0])
    theta4_values.append(solution[1])
    omega3_values.append(solution[2])
    omega4_values.append(solution[3])
    alpha3_values.append(solution[4])
    alpha4_values.append(solution[5])

    force_solution = equations_force([theta2, *solution], alpha2, omega2)
    forces.append(force_solution)

    initial_guess = solution

if failed_angles:
    print(f'Warning: no valid solution at {len(failed_angles)} of {len(theta2_values)} crank angles '
          f'(from {min(failed_angles):.1f} to {max(failed_angles):.1f} degrees). Results there are not physical.')

# Plot theta3 and theta4 as functions of theta2
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(np.degrees(theta2_values), np.degrees(theta3_values))
plt.title(r'$\theta_3$ as a function of $\theta_2$')
plt.xlabel(r'$\theta_2$ (degrees)')
plt.ylabel(r'$\theta_3$ (degrees)')

plt.subplot(1, 2, 2)
plt.plot(np.degrees(theta2_values), np.degrees(theta4_values))
plt.title(r'$\theta_4$ as a function of $\theta_2$')
plt.xlabel(r'$\the
