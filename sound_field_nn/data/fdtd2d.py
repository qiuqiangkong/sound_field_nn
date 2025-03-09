import numpy as np


class FDTD2D:
    def __init__(self, duration: float = 0.1, verbose=False):

        self.duration = duration
        self.dx = 0.02
        self.dy = 0.02
        self.Nx = 100
        self.Ny = 100
        self.c = 343.
        self.dt = self.dx / self.c / 3
        self.skip_steps = 10
        self.verbose = verbose

    def simulate(self) -> dict:

        steps = int(self.duration / self.dt)

        # Check CFL condition for stability
        cfl = self.c * self.dt * np.sqrt(1 / self.dx**2 + 1 / self.dy**2)
        if self.verbose: 
            print(f"CFL number: {cfl:.3f}")
        if cfl > 1.0:
            print("Warning: Simulation may be unstable!")

        # Initialize displacement arrays
        u_prev = np.zeros((self.Nx, self.Ny))
        u_current = np.zeros((self.Nx, self.Ny))
        u_next = np.zeros((self.Nx, self.Ny))

        # Create initial Gaussian pulse
        u_current = self.initial_condition()  # shape: (Nx, Ny)
        buffer = [u_current.copy()]

        # Simulate
        for n in range(steps):
            if self.verbose:
                print("{}/{}".format(n, steps))
            
            laplacian = (
                np.roll(u_current, shift=1, axis=0) + np.roll(u_current, shift=-1, axis=0) +
                np.roll(u_current, shift=1, axis=1) + np.roll(u_current, shift=-1, axis=1) -
                4 * u_current
            )

            u_next[1:-1, 1:-1] = (
                2 * u_current[1:-1, 1:-1] - u_prev[1:-1, 1:-1] +
                (self.c**2 * self.dt**2 / self.dx**2) * laplacian[1:-1, 1:-1]
            )

            # Apply fixed boundary conditions
            u_next[0, :] = u_next[-1, :] = 0
            u_next[:, 0] = u_next[:, -1] = 0
            
            # Update arrays for next iteration
            u_prev[:, :] = u_current
            u_current[:, :] = u_next
            
            buffer.append(u_current.copy())

        buffer = np.array(buffer[0 :: self.skip_steps])  # shape: (t, x ,y)
        time_steps = np.arange(0, steps, self.skip_steps)

        data = {
            "x_init": buffer[0].astype(np.float32),
            "x": buffer.astype(np.float32),
            "t_index": time_steps
        }

        return data

    def initial_condition(self) -> np.ndarray:
        r"""Init u(x, y) at t = 0"""

        cx = self.Nx // 2  # Center position
        cy = self.Ny // 2  # Center position
        x = np.arange(self.Nx)[:, None]
        y = np.arange(self.Ny)[None, :]
        sigma = 1.
        u_current = np.exp(-((x - cx)**2 + (y - cy)**2) / (2*sigma**2))
        return u_current


def visualize(x: np.ndarray) -> None:
    
    fig, ax = plt.subplots()
    cax = ax.imshow(x[0], cmap='viridis', vmin=-1, vmax=1)
    fig.colorbar(cax)

    def update(frame):
        cax.set_data(frame)
        return cax,

    ani = animation.FuncAnimation(fig, func=update, frames=x[0:100], interval=20)
    writer = animation.FFMpegWriter(fps=24)

    out_path = "_zz.mp4"
    ani.save(out_path, writer=writer)
    print("Write sound filed MP4 to {}".format(out_path))


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # FDTD simulator
    sim = FDTD2D(duration=0.1, verbose=True)

    # Simulate
    data = sim.simulate()

    # Print
    x = data["x"]  # (t, Nx, Ny)
    print("Sound field shape: {}".format(x.shape))

    # Write video for visualization
    visualize(x)