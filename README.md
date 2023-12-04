# Semi-supervised ROI tracking for suite2p

A python toolbox for tracking ROIs between <a href="https://github.com/MouseLand/suite2p">suite2p</a> processed two-photon calcium imaging experiments.<br>

Features:<br>
- Manual and automatic identification of matching ROIs<br>
- Alignment of an arbitrary number of planes to a reference plane (e.g., when ROIs in a reference plane are split across two or more planes in a subsequent recording) with conflicting ROI matches resolved automatically.<br>
- ROI match curation<br>

Requires:<br>
<a href="https://github.com/sfailor/key-point-finder">key-point-finder</a><br>
<a href="https://pypi.org/project/opencv-python/">opencv-python</a><br>
pandas<br>
scipy<br>
matplotlib<br>
<a href="https://www.pyqtgraph.org/">pyqtgraph</a><br>
<a href="https://pypi.org/project/PyQt5/">PyQT5</a><br>



## Feature/ROI selection

https://github.com/sfailor/srt4s2p/assets/15325939/a2c99ede-f8e1-4eeb-8060-8ad9759de1ea

## ROI match curation

https://github.com/sfailor/srt4s2p/assets/15325939/332a01f2-9a18-473a-adf7-5905c3e42ac7

<br><br>Inspired by <a href="https://github.com/ransona/ROIMatchPub">ROIMatchPub</a>
