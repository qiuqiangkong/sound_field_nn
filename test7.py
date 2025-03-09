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

# Initialize wave field: u[:,:,0] = previous, u[:,:,1] = current, u[:,:,2] = next
u = np.zeros((nx, ny, 3))

# Define a user-defined boundary shape (e.g., a circular region where waves can propagate)
boundary_mask = np.ones((nx, ny), dtype=bool)  # 1 = wave allowed, 0 = boundary
cx, cy, radius = nx // 2, ny // 2, 30  # Center and radius of the circular boundary
for i in range(nx):
    for j in range(ny):
        if np.sqrt((i - cx) ** 2 + (j - cy) ** 2) > radius:
            boundary_mask[i, j] = 0  # Outside the circle is a boundary

# Initial Condition: Gaussian pulse inside the boundary
sigma = 5.0
for i in range(nx):
    for j in range(ny):
        if boundary_mask[i, j]:  # Only set values inside the allowed region
            u[i, j, 1] = np.exp(-((i - cx)**2 + (j - cy)**2) / (2 * sigma**2))

# Setup figure
fig, ax = plt.subplots()
im = ax.imshow(u[:, :, 1], cmap='RdBu', vmin=-0.1, vmax=0.1)
plt.colorbar(im)
ax.set_title("2D FDTD with User-Defined Boundary")

# Update function for animation
def update(frame):
    global u
    u_next = np.copy(u[:, :, 1])  # Initialize next time step

    # Compute FDTD update only inside the allowed region
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            if boundary_mask[i, j]:  # Update only in valid regions
                u_next[i, j] = (2 * u[i, j, 1] - u[i, j, 0] +
                                r**2 * (u[i + 1, j, 1] + u[i - 1, j, 1] +
                                        u[i, j + 1, 1] + u[i, j - 1, 1] -
                                        4 * u[i, j, 1]))

    # Apply reflecting boundary condition at custom shape (zero velocity at boundary)
    u_next[~boundary_mask] = 0  

    # Rotate time indices
    u[:, :, 0] = u[:, :, 1]
    u[:, :, 1] = u_next
    im.set_array(u[:, :, 1])
    return im,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=nt, interval=30, blit=False)

# Save to MP4
output_filename = "fdtd_simulation2.mp4"
writer = animation.FFMpegWriter(fps=30, bitrate=1800)
ani.save(output_filename, writer=writer)

print(f"Simulation saved as {output_filename}")

# Show animation (optional)
plt.show()
