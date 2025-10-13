import sys
sys.path.append('src')
import enex_analysis as enex
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import numpy as np
import dartwork_mpl as dm
warnings.filterwarnings("ignore")

## Fontsize 지정
plt.rcParams['font.size'] = 9

fs = {
    'label': dm.fs(0),
    'tick': dm.fs(-1.5),
    'legend': dm.fs(-2.0),
    'subtitle': dm.fs(-0.5),
    'cbar_tick': dm.fs(-2.0),
    'cbar_label': dm.fs(-2.0),
    'cbar_title': dm.fs(-1),
    'setpoint': dm.fs(-1),
    'text': dm.fs(-3.0),
            }

pad = {
    'label': 6,
    'tick': 5,
}

LW = np.arange(0.25, 3.0, 0.25)