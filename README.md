# Team4

M1 - Introduction to Human and Computer Vision

## Week 1

### Task 1 - Image retrieval

To execute the program:

```
$ python3 run.py [-b background][-m mode] [-k k] [-c color_space] [-d distance] [-p path_of_the_BBDD] [-q path_of_the_query_set]
```
```
CBIR with different descriptors and distances
arguments:

 -h, --help Show this help message and exit
 -b B       Substract the background of the paintings (y) or not (n)
 -m M       Define if the query set is for developement(d) ot test(t)
 -k K       Number of images to retrieve
 -c C       Color space in which the histograms will be computed
 -d D       Distance/s to compare the histograms
 -p P       Path to the database directory
 -q Q       Path to the query set directory
```
```
Argument options:
 -b:    y - If the images contain a part of the background and it has to be estimated and substracted. The masks of all the query images are stored in a folder called masks.
        n - The images are cut to the frame of the painting
        
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
        
 -d:    euclidean - Euclidean distance
        intersec  - Intersectoin of the histograms
        l1        - L1/Manhattan distance
        chi2      - Chi-squared distance
        hellinger - Hellinger distance
        all       - Compute all the distances (only in development mode)
        
 -p:    Path
 
 -q:    Path
```
Examples:

The images have background, the program is in development mode, only 1 image is retrieved, the histograms are computed using the YCrCb color space and the K images will be computed using all the possible distances:

```
python3 run.py -b y -m d -k 1 -c YCrCb -d all -p /home/Team4/Desktop/M1/data/BBDD/ -q /home/Team4/Desktop/M1/data/qsd1_w1/

Estimating and substracting the background for every query image...
100%|███████████████████████████████████████████| 30/30 [00:25<00:00,  1.16it/s]

BACKGROUND SUBSTRACTION MEASURES:
Precision: 0.9860
Recall: 0.9754
F1-measure: 0.9803

Computing the histograms of all the images of the database...
100%|█████████████████████████████████████████| 287/287 [00:11<00:00, 24.01it/s]

Computing the distances between histograms...
100%|███████████████████████████████████████████| 30/30 [00:57<00:00,  1.93s/it]

mAP@k (K = 1) of the desired distances
Euclidean Distance: 0.4333
Histogram Intersection: 0.3667
L1 Distance: 0.5667
Chi-Squared Distance: 0.5333
Hellinger Distance: 0.5000

```

