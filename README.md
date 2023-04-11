# Semi-supervised ROI tracking for suite2p (srt4s2p)

A python toolbox for tracking ROIs between <a href="https://github.com/MouseLand/suite2p">suite2p</a> processed two-photon calcium imaging experiments.<br>

Features:<br>
- Manual selection of matching ROIs<br>
- Projective or similarity transformations for plane alignment<br>
- Alignment of an arbitrary number of planes to a reference plane (e.g., when ROIs in a reference plane are split across two or more planes in a subsequent recording)<br>
- ROI match curation<br>

Requires:<br>
<a href="https://github.com/sfailor/key-point-finder">key-point-finder</a><br>
<a href="https://pypi.org/project/opencv-python/">opencv-python</a><br>
pandas<br>
scipy<br>
matplotlib<br>
<a href="https://www.pyqtgraph.org/">pyqtgraph</a><br>
<a href="https://pypi.org/project/PyQt5/">PyQT5</a><br>
<br>
Inspired by <a href="https://github.com/ransona/ROIMatchPub">ROIMatchPub</a><br>
