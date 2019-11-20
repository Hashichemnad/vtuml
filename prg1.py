import csv
a=[]
with open('findS.csv') as csfile:
    reader = csv.reader(csfile)
    for row in reader:
        a.append(row)
    print(row)
num_attributes = len(a[0])-1
print("The most general hypothesis:",["?"]*num_attributes)
print("The most specific hypothesis:",["0"]*num_attributes)
hypothesis = a[1][:-1]
print("\nfind S: Finding a maximally specific hypothesis")
for i in range(len(a)):
    if a[i][num_attributes]=="yes":
        for j in range(num_attributes):
            if a[i][j]!=hypothesis[j]:
                hypothesis[j]='?'
        print("h",i,"=",hypothesis[1:])
print("\nThe maximally specific hypothesis for training set is")
print(hypothesis[1:])