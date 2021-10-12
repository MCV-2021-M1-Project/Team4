# Team4

M1 - Introduction to Human and Computer Vision

## Week 1

### Task 1 - Image retrieval

To execute the program:

```
$ python3 run.py [-m mode] [-k k] [-c color_space] [-d distance] [-p path_of_the_BBDD] [-q path_of_the_query_set]
```
```
CBIR with different descriptors and distances
arguments:

 -h, --help Show this help message and exit
 -m M       Define if the query set is for developement(d) ot test(t)
 -k K       Number of images to retrieve
 -c C       Color space in which the histograms will be computed
 -d D       Distance/s to compare the histograms
 -p P       Path to the database directory
 -q Q       Path to the query set directory
```
```
Argument options:
 -m:    d - If the query set is for development and the correspondences with the database are known. The distance (-d) is computed.
        t - If the query set is for test. The program saves a pickle file with a list of list with the K best image numbers in the current directory.
 -k:    int - number of images to retrieve or compute the mAP@k
 -c:    GRAY   - Gray level histograms
        RGB    - RGB histograms
        H      - Hue histogram
        S      - Saturation histogram
        V      - Value histogram
        HS     - HS histograms
        HV     - HV histograms
        HSV    - HSV histograms
        YCrCb  - YCrCb histograms
        CrCb   - CrCb histograms
        CIELab - CIELab histograms
 -d:    euclidean - euclidean distance
        intersec  - intersectoin of the histograms
        l1        - l1/Manhattan distance
        chi2      - Chi-squared distance
        hellinger - Hellinger distance
        all       - Compute all the distances (only in development mode)
 -p:    Path
 -q:    Path
```
Example:
```
python3 run.py -m d -k 1 -c YCrCb -d all -p /home/Team4/Desktop/M1/data/BBDD/ -q /home/Team4/Desktop/M1/data/qsd1_w1/
```

