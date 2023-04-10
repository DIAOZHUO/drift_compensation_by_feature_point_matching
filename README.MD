Drift Compensation By Feature Point Matching
===================================
An algorithm to automatically compensate drfit (thermal drift/piezo creep or so on) from sequence images.
<br>
It can be implemented in transmission electron microscopy, scanning electron microscopy, scanning probe microscopy, and etc...


Example (text.py)
-------
These 3 images (A,B,C) is acquired continuously by scanning tunneling microscopy.
Pixels in image shifts to right due to drift.
<img src="./Figure_1.png" alt="figure1">


In the text.py script, the shift in A $\rightarrow$ B and A $\rightarrow$ C are calculted by feature point matching.
Also, this algorithm contains a selection algorithm to get rid of bad result (red line).

<p>
<img src="./108_20220124_si110_drift.pkl_1.jpg" alt="figure21" width="400">
<img src="./109_20220124_si110_drift.pkl_1.jpg" alt="figure22" width="400">
</p>

Finally, the drift speed can be calculated (Voltage/s).
These values can be fed to a scan instrument as FeedForward signals to cancel drift.

<img src="./Figure_2.png" alt="figure2">

Usage
-------
Compare two images: this code will return the pixel shift.
```python
import cv2
from scripts.feature_matching import feature_matching_map
map1 = cv2.imread("001.jpg")
map2 = cv2.imread("002.jpg")
print(feature_matching_map(map1, map2, draw_plot=True))

```

Compare scan datas with unit: see "text.py"

Installation
-------

Python version 3.* required.
<br>
Please install by the following command via pip:

(our scan data IO and data processing package)
* `pip install SPMUtil`

(some common packages)
* `pip install matplotlib, opencv-python, opencv-contrib-python`

Citation
-------
We will be appreciated if you can cite our work!

Zhuo Diao, Keiichi Ueda, Linfeng Hou, Hayato Yamashita, Oscar Custance, Masayuki Abe, Appl. Phys. Lett. 122, 121601 (2023); https://doi.org/10.1063/5.0139330