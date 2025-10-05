plt.figure()
plt.hist(df["c1"])
plt.show

x1=df["c1"]
x2=["c2"]
sns.scatterplot(x=x1,y=x2,hue["TARGET"])
plt.show()

sns.distplot(df["c2"])
plt.show()

sns.boxplot(df["c1"])
plt.show()

sns.heatmap(data=df.corr(),annot=True)
plt.show()

