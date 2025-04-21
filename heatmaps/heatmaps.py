import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm

from sports.annotators.soccer import draw_pitch
from sports.configs.soccer import SoccerPitchConfiguration

import pathlib
import sys
sys.path.append('../')

from utils import player_tracks_by_ids, remove_short_tracks, get_transformed

def generate_heatmaps(video_name, tracks):

    player_tracks = player_tracks_by_ids(tracks)

    frame = draw_pitch(SoccerPitchConfiguration())

    for id, track in tqdm(player_tracks.items(), desc="Generating heatmaps"):
        positions = get_transformed(track)

        x = np.array([pos[0] for pos in positions])
        y = np.array([pos[1] for pos in positions])

        x = (x*0.1)+50
        y = (y*0.1)+50

        k = gaussian_kde(np.vstack([x, y]))
        xi, yi = np.mgrid[0:1300:x.size**0.5*1j,0:800:y.size**0.5*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        fig = plt.figure(figsize=(7,8))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        # alpha=0.5 will make the plots semitransparent
        ax1.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=0.5)
        ax2.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.5)

        h,w,_ = frame.shape

        ax1.set_xlim(0, w)
        ax1.set_ylim(h, 0)
        ax2.set_xlim(0, w)
        ax2.set_ylim(h, 0)

        im = frame.copy()
        ax1.imshow(im, extent=[0, w, 0, h], aspect='auto')
        ax2.imshow(im, extent=[0, w, 0, h], aspect='auto')

        pathlib.Path(f'output/{video_name}/heatmaps').mkdir(parents=True, exist_ok=True)

        extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f'output/{video_name}/heatmaps/ID - {id}.png', bbox_inches=extent)

        plt.close(fig)

                