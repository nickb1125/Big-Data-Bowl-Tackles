# Spatial Density Estimation for Tackle Success Using Deep Convolutional Gauccian-Mixture Ensembling 
#### Expected Yards Saved (EYS) & Percent Field Influence (PFI): Metrics for Tackler Efficiencies & Role Classification

# Main Goals

* **A shortcoming of the current state of sports analytics is failing to recognize the limitations of our data and the degree of prediction confidence we may have.** Limited data means limited and **varying confidence depending on the in-play situation**, and we should account and report for these varying intervals. From play-call-decision analytics (see Ryan Brill's talk here) to  metrics like those in the BDB, we should start reporting metric confidence. **We estimate variance in prediction by using ensemble model methods for our spatial densities.**

* Tackling encompasses more than big hits: **valuable tacking skills include any type of coersion that reduces the ball carriers yardage** by the end of the play. This can be direct, like the highlight hits you'll see on replay, or indirect like having positioning that manipulates the ball carrier into a worse route or out of bounds. We should have ways of measuring how direct a defenders influence is spatially, and how much he reduces yardage.

* **Spatial density estimation and the use of the subtraction method allows us to (1) where the tackler is coercing the ball carrier to go in the context of the rest of his team, and (2) how direct, or broad, his influence on the ball carrier is.**


# Code Map

```
<div align="center">
python 001_preprocess.py
</div>
```