"""

    
Least squares is a method to find the best-fitting line by minimizing the total error between the predicted values and the actual values.

    The error for each data point is the difference between the actual yy and the predicted y^y^​ from the line.

    We square these errors to avoid negative and positive errors canceling out.

    Then, we sum all the squared errors for all data points.

Mathematically, for data points (xi,yi)(xi​,yi​) where i=1,2,...,ni=1,2,...,n, we want to minimize:
S=∑i=1n(yi−yi^)2=∑i=1n(yi−(mxi+b))2
S=i=1∑n​(yi​−yi​^​)2=i=1∑n​(yi​−(mxi​+b))2

Why is least squares used?

    It gives a unique, simple way to find the line that best represents the trend in the data.

    Squaring emphasizes larger errors more strongly, making the fit sensitive to big deviations.

    It has nice mathematical properties, making it easy to solve analytically.
    
"""