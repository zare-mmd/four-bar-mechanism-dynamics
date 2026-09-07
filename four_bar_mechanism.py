# Mohammad Zare
# Dynamics of Machinery (DOM) project
# Student ID: 40030570
#
# Kinematic and dynamic force analysis of a four-bar mechanism.
# Solves position, velocity and acceleration equations for the coupler
# and rocker links, then computes joint reaction forces for each link
# using a Newton-Euler style force balance. Includes an animation of
# the mechanism over one full revolution of the crank.

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
    eq3 = l4 * omega4 * np.sin(theta4) - l3 * omega3 * np.sin(theta3) - l2 * omega2 ** 2 * np.sin(theta2)
    eq4 = l2 * omega2 ** 2 * np.cos(theta2) - l4 * omega4 * np.cos(theta4) + l3 * omega3 * np.cos(theta3)

    # Acceleration
    eq5 = (l2 * alpha2 * np.cos(theta2) - l2 * omega2 * np.sin(theta2)
           + l3 * alpha3 * np.cos(theta3) - l3 * omega3 ** 2 * np.sin(theta3)
           - l4 * alpha4 * np.cos(theta4) + l4 * omega4 ** 2 * np.sin(theta4))
    eq6 = (l2 * alpha2 * np.sin(theta2) - l2 * omega2 ** 2 * np.cos(theta2)
           - l3 * alpha3 * np.sin(theta3) - l3 * omega3 ** 2 * np.cos(theta3)
           + l4 * alpha4 * np.sin(theta4) + l4 * omega4 ** 2 * np.cos(theta4))

    return [eq1, eq2, eq3, eq4, eq5, eq6]


def equations_force(vars, alpha2, omega2):
    """Compute joint reaction forces on each link from its kinematic state."""
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

# Initial guess for [theta3, theta4, omega3, omega4, alpha3, alpha4]
initial_guess = [0, 0, 0, 0, 0, 0]

# Solve the equations for each crank angle, using the previous solution
# as the initial guess for the next angle (continuation method)
for theta2 in theta2_values:
    solution = fsolve(equations_position, initial_guess, args=(theta2))
    theta3_values.append(solution[0])
    theta4_values.append(solution[1])
    omega3_values.append(solution[2])
    omega4_values.append(solution[3])
    alpha3_values.append(solution[4])
    alpha4_values.append(solution[5])

    force_solution = equations_force([theta2, *solution], alpha2, omega2)
    forces.append(force_solution)

    initial_guess = solution

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
plt.xlabel(r'$\theta_2$ (degrees)')
plt.ylabel(r'$\theta_4$ (degrees)')
plt.tight_layout()
plt.show()

# Plot omega3 and omega4 as functions of theta2
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(np.degrees(theta2_values), np.degrees(omega3_values))
plt.title(r'$\omega_3$ as a function of $\theta_2$')
plt.xlabel(r'$\theta_2$ (degrees)')
plt.ylabel(r'$\omega_3$ (rad/s)')

plt.subplot(1, 2, 2)
plt.plot(np.degrees(theta2_values), np.degrees(omega4_values))
plt.title(r'$\omega_4$ as a function of $\theta_2$')
plt.xlabel(r'$\theta_2$ (degrees)')
plt.ylabel(r'$\omega_4$ (rad/s)')
plt.tight_layout()
plt.show()

# Plot alpha3 and alpha4 as functions of theta2
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(np.degrees(theta2_values), np.degrees(alpha3_values))
plt.title(r'$\alpha_3$ as a function of $\theta_2$')
plt.xlabel(r'$\theta_2$ (degrees)')
plt.ylabel(r'$\alpha_3$ (rad/s^2)')

plt.subplot(1, 2, 2)
plt.plot(np.degrees(theta2_values), np.degrees(alpha4_values))
plt.title(r'$\alpha_4$ as a function of $\theta_2$')
plt.xlabel(r'$\theta_2$ (degrees)')
plt.ylabel(r'$\alpha_4$ (rad/s^2)')
plt.tight_layout()
plt.show()

# Plot forces in X and Y directions for each link
forces = np.array(forces)
plt.figure(figsize=(8, 6))
plt.plot(np.degrees(theta2_values), forces[:, ::2])
plt.plot(np.degrees(theta2_values), forces[:, 1::2])
plt.title('Forces in X and Y directions for each link')
plt.xlabel(r'$\theta_2$ (degrees)')
plt.ylabel('Forces')
plt.legend(['F2_x', 'F3_x', 'F4_x', 'F2_y', 'F3_y', 'F4_y'])
plt.show()

# Plot total (resultant) force magnitude for each link
plt.figure(figsize=(8, 6))
plt.plot(np.degrees(theta2_values), forces[:, 6:])
plt.title('Total forces for each link')
plt.xlabel(r'$\theta_2$ (degrees)')
plt.ylabel('Total forces')
plt.legend(['F2', 'F3', 'F4'])
plt.show()


def calculate_positions(theta2, solution):
    """Compute the (x, y) positions of each joint for a given crank angle."""
    x0, y0 = 0, 0
    x1, y1 = x0 + l1 * np.cos(theta2), y0 + l1 * np.sin(theta2)
    x2, y2 = x1 + l2 * np.cos(solution[0]), y1 + l2 * np.sin(solution[0])
    x3, y3 = x2 + l3 * np.cos(solution[1]), y2 + l3 * np.sin(solution[1])
    x4, y4 = x3 + l4 * np.cos(solution[2]), y3 + l4 * np.sin(solution[2])
    return x0, y0, x1, y1, x2, y2, x3, y3, x4, y4


def update(frame):
    """Animation update function: redraw the mechanism at the given frame's crank angle."""
    plt.clf()
    theta2 = theta2_values[frame]
    solution = fsolve(equations_position, initial_guess, args=(theta2))
    x0, y0, x1, y1, x2, y2, x3, y3, x4, y4 = calculate_positions(theta2, solution)

    plt.plot([x0, x1], [y0, y1], color='b', linewidth=2, marker='o')
    plt.plot([x1, x2], [y1, y2], color='r', linewidth=2, marker='o')
    plt.plot([x2, x3], [y2, y3], color='g', linewidth=2, marker='o')
    plt.plot([x3, x4], [y3, y4], color='m', linewidth=2, marker='o')
    plt.scatter([x0, x1, x2, x3, x4], [y0, y1, y2, y3, y4], color='black')

    plt.title('Animation of Links')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')


fig = plt.figure(figsize=(8, 8))
animation = FuncAnimation(fig, update, frames=len(theta2_values), interval=50)
plt.show()
