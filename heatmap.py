#df is normalised data frame#######
correlation = df.corr(method="pearson")
plt.rcParams['figure.figsize'] = (15,15)
sb.heatmap(correlation,xticklabels=correlation.columns,yticklabels=correlation.columns,cmap="RdBu_r",linewidth=0.5,annot=True)
