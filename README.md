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

https://user-images.githubusercontent.com/15325939/232312635-e4278c50-e28a-4fbf-9a0b-161dc14f12ec.mp4 


## ROI match curation

https://user-images.githubusercontent.com/15325939/232312750-ec15e66f-a2a4-45b6-a6dc-06d2422dd02b.mp4 

<br><br>Inspired by <a href="https://github.com/ransona/ROIMatchPub">ROIMatchPub</a>
