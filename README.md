machi.py: marching code visualization

use --csv (file name) to import csv
--csv sample_wifi_data.csv
csv must have 4 columns; x, y, z, and metric
leave empty to visualize randomly generated one with 1 ~ 8 router visualized

use --levels (positive boundary: green) (negative boundary: red) to adjust boundaries
default is --levels 60 20

input (x cord) (y cord) (z cord) (metric) to update the map
it will change the most near existing point's metric and update the visualization result

wifi metric is (and should be) normalized to have 100 as maximum wifi strength

sample_wifi_data.csv: one of randomly generated data, as csv form

machi_sample.py
code used when generating sample_wifi_data.csv
number of router is fixed as 4 for better visualization
