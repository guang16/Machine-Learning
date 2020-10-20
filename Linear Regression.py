import numpy

# import the CSV and parse it to Martix type.
Matrix = numpy.loadtxt(open("winequality-white.csv", "rb"), delimiter=";", skiprows=1)

print(Matrix)
