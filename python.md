# Python

## AI

### $Machine \ Learning$

$numpy$ 常用函数

```python
np.shape, np.shape[x], np.dtype, np.size, np.ndim, np.T, np.itemsize, np.nbytes, np.flat
np.array([1, 2, 3, 4, 5])
np.arange(0, 5, 1) #[0, 1, 2, 3, 4] 
np.ones(2)
np.zeros(2)
np.ones_like(ary)
np.zeros_like(ary)
np.empty((x, y, z)) #未初始化的空数组
np.linspace(-np.pi, np.pi, 200) #等差
np.logspace(st, ed, cnt, base) #等比
bry = ary.reshape(x, y), ary.ravel() #视图变维
ary.flatten() #复制变维
ary.shape = (x, y), ary.resize(x, y) #就地变维
ary.astype('float32'), ary.astype('datatime64[Y/M/D/h/m/s]')
ary[x, y, z] = ary[x][y][z] #索引
ary.mean()
ary['f0'] #列 
np.array(data, dtype = {'names': ['name', 'score', 'age'], 'formats': ['U3', '3int32', 'int32']}) #对列重命名
# 索引会降维，而切片不会降维
# 掩码操作：布尔掩码，索引掩码 ary[(ary % 3 == 0) & (ary % 7 == 0)]
c = np.vstack(a, b); d, e = np.vsplit(c, cnt) #垂直方向 
c = np.hstack(a, b); d, e = np.hsplit(c, cnt) #水平方向
c = np.dstack(a, b); d, e = np.dsplit(c, cnt) #深度方向（3维) (x, y)->(x, y, 0)
np.concatenate((a, b), axis=0)
np.split(c, 2, axis=0)
# axis: 0垂直方向组合，1水平方向组合，2深度方向组合
ary.tolist()
np.average(ary, weights=volumes) #加权平均值
np.random.randint(x, y, cnt) #[x, y)之间cnt个随机数
np.max(ary), np.min(ary), np.ptp(ary) #最大值/最小值/极差
np.argmax(ary), np.argmin(ary) #最大/最小元素的下标
np.maximum(a, b), np.minimum(a, b) #两个同维数组中对应元素最大/最小元素构成一个新的数组
np.median(ary) #中位数
np.std(ary) #标准差
```

pandas

Series 带有自定义索引的一维数组

DataFrame 带有自定义索引的二维数组

```python
pd.Series(data, index=[]) #两套索引：标签索引，位置索引
# 标签索引的切片终止位置也包含，即[st, ed]而非[st, ed)
# Series中没有反向索引，但设置标签索引后则可以取到最后一个位置
# DataFrame列与列中的数据格式可以不同
pd.DataFrame(data, index=[], columns=[]) #行级、列级索引
dic= {'Name': pd.Series([], index=), 'Age': pd.Series([], index=)}
pd.DataFrame(dic)
df.pop('column') #删除一列
df.drop(['column1', 'column2'], axis=1, inplace=True) #删除多列，索引需要从水平方向上找
df.drop(['index1', 'index2'], axis=0, inplace=True) #删除多行，索引需要从垂直方向上找
# 访问一行数据 loc标签索引 iloc位置索引
df.loc['index1']
df1 = df1.append(df2) #行的添加
df.mean(axis=0)
df.idxmax(), df.idxmin() #最大/最小元素的下标，返回的是标签索引
df.std() #标准差
df.map({'男':1, '女':0}) #把数据集中某列的所有元素进行替换
df.apply(func)
df.rename(columns={'x':'y'}, inplace=True) #列重命名
df[column].map(dic) #映射
df.drop(df[df[column].apply()].index) #删除某列中满足条件的行
```



### $img\_segmentation \ note$

$Histogram \ Equalization$ 直方图均衡化，被认为是提升图像对比度最为有效的方法，它的基本思想是用数学方法重新调整像素的亮度分布，使调整后的直方图具有最大的动态范围，每个桶 $(bin/bucket)$ 盛纳的像素数量几乎相等。

归一化：$p_r(r_k) = \frac{n_k}{H \times W}, k = 0,1,\dots ,L - 1$     $n_k$ 为灰度值 $r_k$ 的像素的出现的次数

映射函数：$s_k = T(r_k) = (L - 1) \sum_{j=0}^{k}p_r(r_j)$       $s_k$ 为变换后的灰度值



伽马变换（幂律变换）是常用的灰度变换，是一种简单的图像增强算法    $s = cr^{\gamma}$ 

$r$ 为灰度图像的输入值（原来的灰度值），取值范围为 $[0,1]$ 

$s$ 为经过伽马变换后的灰度输出值。$c$ 为灰度缩放系数，通常取 $1$

$\gamma$ 为伽马因子大小。控制了整个变换的缩放程度



高斯平滑是一种图像平滑技术，通过应用高斯滤波器对图像进行滤波来减少图像中的噪声和细节

```python
cv2.equalizeHist
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
img_dst_clahe = clahe.apply(img_src)
#clipLimit：对比对限制阈值，默认为40
#tileGridSize：直方图均衡的栅格尺寸，输入图像将会按照该尺寸分隔后进行局部直方图均衡，默认是8×8大小
blurred_image = cv2.GaussianBlur(src, kernelsize, sigmaX, sigmaY)
```
