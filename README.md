# Naive Bayes Supervised Classification on 2D Point
The code is using the naive Bayes method to classify 2D tagged data into 2 categories. 

Then code runs on python3:

    python3 naiveBayes_2gausClassification.py

<img src="https://github.com/SphericalCowww/ML_naiveBayes/blob/main/naiveBayes2gaus100_Display.png" width="640" height="480">

The training data are given on the top-left plots in blue/red category; their are generated using Gaussians distribution. These distributions projected to the X-axis and Y-axis as shown on the bottom-left and top-right plots. The projected ditributions are padded by 0.001 (adding 0.001 in each bin) and then normalized. The black stars on the top-left plot are test data point to be categorized. Given a test point (x, y), its score is simply calculated by 

$score = log(prior*P_X(x)*P_Y(y))$

for each category. The prior is given by (number of traning data of the category in question)/(total number of training data). The resulting classification of the test data is then given on the bottom-right plot. The "ambiguous" classification follow 2 conditions. The first is if the difference in score is smaller than 1, and the second is if both P_X(x) and P_Y(y) are at minimum for both blue/red categories.

If we increase the test data points from 100 to 10000, it gives the following plot:

<img src="https://github.com/SphericalCowww/ML_naiveBayes/blob/main/naiveBayes2gaus10000_Display.png" width="640" height="480">

References:
- StatQuest with Josh Starmer's Youtube channel (<a href="https://www.youtube.com/watch?v=O2L2Uv9pdDA">Youtube1</a>, <a href="https://www.youtube.com/watch?v=H3EjCKtlVog">2</a>)
