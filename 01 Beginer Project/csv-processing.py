import csv

array1 = []

with open('S7-1500.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader: {
        array1.append(str(row[2]))
    }

    print(len(array1))
