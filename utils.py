import matplotlib.pyplot as plt
import os

def create_video(trajectories):
    ind = 0
    traj = trajectories[0]
    for x in traj[:90]:
        plt.imshow(x)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig('%d.png' % ind)
        plt.close()
        ind += 1

    os.system("ffmpeg -r 30 -i %d.png -vcodec mpeg4 -y movie.mp4")
    os.system("rm -rf *.png")

