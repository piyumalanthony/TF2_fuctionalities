from matplotlib import pyplot as plt

x = [1,2,3,4,5,6,7,8]
y = [2,4,6,8,3,4,5,1]

plt.scatter(x,y, label = 'skitscat', color ='k', s=25, marker='o')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot')
plt.legend()
plt.show()

