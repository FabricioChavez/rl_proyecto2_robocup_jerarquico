"""Campo adaptado de draw_pitch del notebook de Jeffrey / grupo 02.
Fuente: https://github.com/jeffreymonjacastro/grupo-02-proyecto-rl
Commit: 172f5523910c3be0bf537c9a57f26120aa6ec95f
Cambio: encuadre de campo completo para incluir defensa, arquero y definición.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_pitch(ax, title):
    ax.set_facecolor("#2e7d32")
    pitch_border = patches.Rectangle(
        (-52.5, -34), 105, 68, linewidth=1.8, edgecolor="white", facecolor="none"
    )
    half_line = plt.Line2D([0, 0], [-34, 34], color="white", linewidth=1.8)
    center_circle = patches.Circle(
        (0, 0), 9.15, linewidth=1.8, edgecolor="white", facecolor="none"
    )
    center_spot = patches.Circle((0, 0), 0.5, color="white")
    left_box = patches.Rectangle(
        (-52.5, -20.16), 16.5, 40.32, linewidth=1.2, edgecolor="white", facecolor="none"
    )
    right_box = patches.Rectangle(
        (36.0, -20.16), 16.5, 40.32, linewidth=1.2, edgecolor="white", facecolor="none"
    )
    left_goal = patches.Rectangle(
        (-54.5, -7.0),
        2.0,
        14.0,
        linewidth=1.5,
        edgecolor="yellow",
        facecolor="#ffffff",
        alpha=0.3,
    )
    right_goal = patches.Rectangle(
        (52.5, -7.0),
        2.0,
        14.0,
        linewidth=1.5,
        edgecolor="yellow",
        facecolor="#ffffff",
        alpha=0.3,
    )

    ax.add_patch(pitch_border)
    ax.add_line(half_line)
    ax.add_patch(center_circle)
    ax.add_patch(center_spot)
    ax.add_patch(left_box)
    ax.add_patch(right_box)
    ax.add_patch(left_goal)
    ax.add_patch(right_goal)

    ax.set_xlim(-58, 58)
    ax.set_ylim(-38, 38)
    ax.set_xlabel("Coordenada X (metros)", fontsize=9.5)
    ax.set_ylabel("Coordenada Y (metros)", fontsize=9.5)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
