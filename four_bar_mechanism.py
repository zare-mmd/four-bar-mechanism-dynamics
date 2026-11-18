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
plt.xlabel(r'$\theta_2$ (degrees)')
plt.ylabel(r'$\theta_4$ (degrees)')
plt.tight_layout()
plt.show()

# Plot omega3 and omega4 as functions of theta2
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(np.degrees(theta2_values), omega3_values)
plt.title(r'$\omega_3$ as a function of $\theta_2$')
plt.xlabel(r'$\theta_2$ (degrees)')
plt.ylabel(r'$\omega_3$ (rad/s)')

plt.subplot(1, 2, 2)
plt.plot(np.degrees(theta2_values), omega4_values)
plt.title(r'$\omega_4$ as a function of $\theta_2$')
plt.xlabel(r'$\theta_2$ (degrees)')
plt.ylabel(r'$\omega_4$ (rad/s)')
plt.tight_layout()
plt.show()

# Plot alpha3 and alpha4 as functions of theta2
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(np.degrees(theta2_values), alpha3_values)
plt.title(r'$\alpha_3$ as a function of $\theta_2$')
plt.xlabel(r'$\theta_2$ (degrees)')
plt.ylabel(r'$\alpha_3$ (rad/s^2)')

plt.subplot(1, 2, 2)
plt.plot(np.degrees(theta2_values), alpha4_values)
plt.title(r'$\alpha_4$ as a function of $\theta_2$')
plt.xlabel(r'$\theta_2$ (degrees)')
plt.ylabel(r'$\alpha_4$ (rad/s^2)')
plt.tight_layout()
plt.show()

# Plot forces in X and Y directions for each link
forces = np.array(forces)
plt.figure(figsize=(8, 6))
plt.plot(np.degrees(theta2_values), forces[:, 0:6:2])   # columns 0, 2, 4 -> F2_x, F3_x, F4_x
plt.plot(np.degrees(theta2_values), forces[:, 1:6:2])   # columns 1, 3, 5 -> F2_y, F3_y, F4_y
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


def calculate_positions(theta2, theta3):
    """Compute the (x, y) positions of the joints for a given crank angle.
    O2 = crank pivot, A = crank/coupler joint, B = coupler/rocker joint, O4 = rocker pivot."""
    xO2, yO2 = 0, 0
    xA, yA = l2 * np.cos(theta2), l2 * np.sin(theta2)
    xB, yB = xA + l3 * np.cos(theta3), yA + l3 * np.sin(theta3)
    xO4, yO4 = l1, 0
    return xO2, yO2, xA, yA, xB, yB, xO4, yO4


def update(frame):
    """Animation update function: redraw the mechanism at the given frame's crank angle."""
    plt.clf()
    theta2 = theta2_values[frame]
    theta3 = theta3_values[frame]
    xO2, yO2, xA, yA, xB, yB, xO4, yO4 = calculate_positions(theta2, theta3)

    plt.plot([xO2, xO4], [yO2, yO4], color='k', linewidth=2, linestyle='--')  # ground (link 1)
    plt.plot([xO2, xA], [yO2, yA], color='b', linewidth=2, marker='o')        # crank (link 2)
    plt.plot([xA, xB], [yA, yB], color='r', linewidth=2, marker='o')          # coupler (link 3)
    plt.plot([xB, xO4], [yB, yO4], color='g', linewidth=2, marker='o')        # rocker (link 4)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(x_min, x_max)   # fixed frame, so the view does not jump between frames
    plt.ylim(y_min, y_max)

    plt.title('Animation of Links')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')


# Fixed axis limits for the animation, computed once from all joint positions
xA_all = l2 * np.cos(theta2_values)
yA_all = l2 * np.sin(theta2_values)
xB_all = xA_all + l3 * np.cos(theta3_values)
yB_all = yA_all + l3 * np.sin(theta3_values)
margin = 0.1 * (l1 + l2 + l3 + l4)
x_min = min(0, xA_all.min(), xB_all.min()) - margin
x_max = max(l1, xA_all.max(), xB_all.max()) + margin
y_min = min(0, yA_all.min(), yB_all.min()) - margin
y_max = max(0, yA_all.max(), yB_all.max()) + margin

fig = plt.figure(figsize=(8, 8))
animation = FuncAnimation(fig, update, frames=len(theta2_values), interval=50)
plt.show()
