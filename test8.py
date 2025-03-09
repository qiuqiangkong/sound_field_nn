import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simulation Parameters
nx, ny = 100, 100  # Grid size
nt = 200  # Number of time steps
c = 1.0  # Wave speed
dx = dy = 1.0  # Spatial step
dt = 0.5  # Time step
r = (c * dt / dx)  # CFL condition (must be <= 1/sqrt(2))

# Initialize wave field
u = np.zeros((nx, ny, 3))  # u[i, j, n] where n=0,1,2 (current, previous, next)

# Initial condition: Gaussian pulse at center
x0, y0 = nx // 2, ny // 2
sigma = 5.0
for i in range(nx):
    for j in range(ny):
        u[i, j, 1] = np.exp(-((i - x0)**2 + (j - y0)**2) / (2 * sigma**2))

# Setup figure
fig, ax = plt.subplots()
im = ax.imshow(u[:, :, 1], cmap='RdBu', vmin=-0.1, vmax=0.1)
plt.colorbar(im)

# Update function for animation
def update(frame):
    global u
    u[1:-1, 1:-1, 2] = (2 * u[1:-1, 1:-1, 1] - u[1:-1, 1:-1, 0] +
                         r**2 * (u[2:, 1:-1, 1] + u[:-2, 1:-1, 1] +
                                 u[1:-1, 2:, 1] + u[1:-1, :-2, 1] -
                                 4 * u[1:-1, 1:-1, 1]))

    # Rotate time indices
    u[:, :, 0] = u[:, :, 1]
    u[:, :, 1] = u[:, :, 2]

    im.set_array(u[:, :, 1])
    return im,

# Save animation to video
fps = 30  # Frames per second
output_filename = "fdtd_simulation.mp4"

writer = animation.FFMpegWriter(fps=fps)
ani = animation.FuncAnimation(fig, update, frames=nt, interval=30, blit=True)

print(f"Saving animation to {output_filename}...")
ani.save(output_filename, writer=writer)
print("Video saved successfully!")

plt.show()
