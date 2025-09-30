import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider


a0 = 5                 # Bohr radius (arbitrary units)
grid_points = 60        # grid resolution
threshold_1s = 0.0005
threshold_2s = 0.0003
num_orbit_points = 200
frames = 300
Z = 3                  # Lithium atomic number


r = np.linspace(-20, 20, grid_points)
X, Y, Z_grid = np.meshgrid(r, r, r)
R = np.sqrt(X**2 + Y**2 + Z_grid**2)


def orbital_1s(r):
    return (1/np.pi/a0**3)**0.5 * np.exp(-r / a0)

def orbital_2s(r):
    return (1/(4*np.sqrt(2*np.pi*a0**3))) * (2 - r/a0) * np.exp(-r/(2*a0))

density_1s = orbital_1s(R)**2
density_2s = orbital_2s(R)**2

mask_1s = density_1s > threshold_1s
mask_2s = density_2s > threshold_2s


orbit_radii = [a0, a0, 2*a0]  # 2 electrons in 1s, 1 in 2s
theta_orbit = np.linspace(0, 2*np.pi, num_orbit_points)
phases = [0, np.pi, 0]  # 1s electrons offset

x_orbits = [r*np.cos(theta_orbit + phases[i]) for i, r in enumerate(orbit_radii)]
y_orbits = [r*np.sin(theta_orbit + phases[i]) for i, r in enumerate(orbit_radii)]
z_orbits = [np.zeros_like(theta_orbit) for _ in orbit_radii]


E_1s = -13.6 * Z**2 / 1**2  # eV
E_2s = -13.6 * Z**2 / 2**2  # eV


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1,1,1])


if np.any(mask_1s):
    sizes_1s = 2 + 20*density_1s[mask_1s]/np.max(density_1s[mask_1s])
    sc1 = ax.scatter(X[mask_1s], Y[mask_1s], Z_grid[mask_1s],
                     c=density_1s[mask_1s], cmap='viridis',
                     s=sizes_1s, alpha=0.7, label='1s orbital')

if np.any(mask_2s):
    sizes_2s = 2 + 20*density_2s[mask_2s]/np.max(density_2s[mask_2s])
    sc2 = ax.scatter(X[mask_2s], Y[mask_2s], Z_grid[mask_2s],
                     c=density_2s[mask_2s], cmap='plasma',
                     s=sizes_2s, alpha=0.5, label='2s orbital')


colors = ['blue', 'cyan', 'magenta']
for i in range(3):
    ax.plot(x_orbits[i], y_orbits[i], z_orbits[i],
            color=colors[i], linewidth=2, label=f'Electron orbit {i+1}')

electrons = [ax.scatter([x_orbits[i][0]], [y_orbits[i][0]], [z_orbits[i][0]],
                        color=colors[i], s=80) for i in range(3)]


ax.set_xlabel('X (a.u.)', fontsize=12)
ax.set_ylabel('Y (a.u.)', fontsize=12)
ax.set_zlabel('Z (a.u.)', fontsize=12)
ax.set_title('3D Lithium Atom: Electron Map of 1s Orbital', fontsize=16)

fig.text(0.5, 0.01, 'Tennessee Technological University - 2025 - Jeffrey R Snyder',
         ha='center', fontsize=10, color='gray')

fig.text(0.5, 0.95, "Purpose: 3D Visualization of Electron Probability Maps with Refinements\n"
                     "Focusing on the 1s Orbital of the Lithium Atom",
         ha='center', fontsize=10, color='black')

ax.text(orbit_radii[0]+1, 0, 0, f"1s: {E_1s:.1f} eV", color='blue', fontsize=10)
ax.text(orbit_radii[2]+1, 0, 0, f"2s: {E_2s:.1f} eV", color='magenta', fontsize=10)


ax_alpha = plt.axes([0.25, 0.02, 0.5, 0.02])
slider_alpha = Slider(ax_alpha, 'Cloud Alpha', 0.0, 1.0, valinit=0.7)

def update(val):
    alpha = slider_alpha.val
    if 'sc1' in globals(): sc1.set_alpha(alpha)
    if 'sc2' in globals(): sc2.set_alpha(alpha*0.7)
    fig.canvas.draw_idle()

slider_alpha.on_changed(update)

def animate(i):
    idx = i % num_orbit_points
    for j, electron in enumerate(electrons):
        electron._offsets3d = ([x_orbits[j][idx]], [y_orbits[j][idx]], [z_orbits[j][idx]])
    return electrons

ani = FuncAnimation(fig, animate, frames=frames, interval=50)

plt.show()
