import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def visualize_trajectory(past_xy,true_future_xy,pred_future_xy,save_path="agent_traj.gif"):
    """
    Create a GIF comparing predicted vs true path.

    past_xy : (T_past, 2)
    true_future_xy : (T_future, 2)
    pred_future_xy : (T_future, 2)
    save_path : str
    """

    title="Agent trajectory"
    past_label="past"
    true_label="true future"
    pred_label="pred future"
    fps=8
    pad=5.0
    figsize=(4, 4)

    past_xy = np.asarray(past_xy)
    true_future_xy = np.asarray(true_future_xy)
    pred_future_xy = np.asarray(pred_future_xy)

    assert past_xy.ndim == 2 and past_xy.shape[1] == 2
    assert true_future_xy.shape == pred_future_xy.shape
    T_future = true_future_xy.shape[0]

    # global bounds so every frame has same window
    all_pts = np.concatenate([past_xy, true_future_xy, pred_future_xy], axis=0)
    xmin, ymin = all_pts.min(axis=0)
    xmax, ymax = all_pts.max(axis=0)

    xmin -= pad
    ymin -= pad
    xmax += pad
    ymax += pad

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)

    # show past
    past_line, = ax.plot(
        past_xy[:, 0],
        past_xy[:, 1],
        "o-",
        color="blue",
        linewidth=1.5,
        markersize=3,
        label=past_label,
    )

    # animated true/pred
    true_line, = ax.plot([], [], "o-", color="green", linewidth=1.5, markersize=3, label=true_label)
    pred_line, = ax.plot([], [], "o--", color="red", linewidth=1.5, markersize=3, label=pred_label)

    # legend
    ax.legend(loc="upper right", fontsize=6, frameon=True)

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    def init():
        true_line.set_data([], [])
        pred_line.set_data([], [])
        return true_line, pred_line

    def update(frame):
        # frame 0..T_future-1
        true_line.set_data(true_future_xy[:frame+1, 0], true_future_xy[:frame+1, 1])
        pred_line.set_data(pred_future_xy[:frame+1, 0], pred_future_xy[:frame+1, 1])
        return true_line, pred_line

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=T_future,
        interval=1000 / fps,
        blit=True,
    )

    anim.save(save_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
