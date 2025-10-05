plt.figure()
plt.hist(df["c1"])
plt.show

sns.histplot(df["c1"], bins=10, kde=True)
plt.show
#use for outlier detection (distributional skewness and extreme vals)
#univariate
#focus on understanding distribution, spread
#find mean,meadian,mode

x1=df["c1"]
x2=["c2"]
sns.scatterplot(x=x1,y=x2,hue["TARGET"])
plt.show()
#use for outlier detection (bivariate)
#bivariate
#correlation (numeric-numeric)
#group difference (numeric-categorical)
#scatter patterns

sns.distplot(df["c2"])
plt.show()

sns.boxplot(df["c1"])
plt.show()
#use for outlier detection (univariate)
#univariate
#focus on understanding distribution, spread
#find mean,meadian,mode

sns.heatmap(data=df.corr(),annot=True)
plt.show()
#Multivariate 3 or more vars
#understand complex interaction

sns.pairplot(df)
plt.show()
#Multivariate 3 or more vars
#understand complex interaction



