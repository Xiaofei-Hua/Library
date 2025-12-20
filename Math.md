# **Math**

****

## 排列

将全排列按字典序排序，若前一个排列为 $\{ p_1, p_2, \dots, p_{k - 1}, p_k, p_{k + 1}, \dots, p_j \dots, p_n \}$ 满足 $p_k < p_{k + 1}, [p_{k + 1}, p_n] 降序，且p_j为最右侧>p_k的一个$ 

则后一个排列为 $\{p_1, p_2, \dots,p_{k - 1},p_j,p_n,p_{n - 1}, \dots, p_{j + 1}, p_k, p_{j - 1}, \dots,p_{k + 1} \}$ ，即$将[p_{k + 1}, p_n]翻转，再将p_k与p_j交换$



所有长度为 $n$ 的全排列的逆序对数目为 $\frac{(n - 1) * (n + 1)!}{6}$ 

排列置换环：每次交换任意两个元素，则环的数量要么加一要么减一

$Wilson$ 定理：对于素数 $p$ 有 $(p-1)! \equiv -1 \ (mod \ p)$ ；对于非素数，若 $p$ 为 $4$ ，则结果为 $2$，否则为 $0$ 

对于一个序列，只能相邻两两交换时，排成有序序列的最小次数为逆序对个数

****

## 整除

### 一些性质

$$
(x, y, z) = (x, y - x, z - y)
$$

$$
若 \ d = (x, y), t = x \ \% \ y \\ 
则 \ t \ \% \ d = 0
$$

$$
构造(a, b) = w，则a = w \times k，b = w \times (k + 1)
$$

$$
\frac{a}{b} \ ((a, b) = 1)是无限循环小数的充要条件是：b包含2和5以外的质因数
$$

### 扩展欧几里得

$$
\because d = exgcd(a, b, x, y) \ 且 \ (x_0, y_0) \ 为一组解
$$

$$
\therefore x' = x_0 \pm k \times b, \quad y' = y_0 \mp k \times a

$$

$$
令 \ t = \lvert \frac{b}{d} \rvert
$$

$$
则最小正整数 x_1 = (x \ \% \ t + t) \ \% \ t ，当\ d\ 需要扩大 \ n \ 倍时依旧成立
$$

$$
所对应的 \ y_1 \ 则通过 \ a \times x_1 + b \times y_1 = d \ 求出
$$



### 中国剩余定理

#### 当模数互素时

$$
令M = \prod_{i = 1}^{n}a_i \\
构造x：令M_j = \frac{M}{a_j},令t_j = M^{-1}(mod \ a_j),则x = \sum_{i = 1}^{n}a_iM_it_i
$$

一条线段上整点的数量 $gcd(abs(x_1-x_2), abs(y_1-y_2)) + 1$ 

若 $(a, n) = 1$ ，那么在 $[0, n)$ 内 $ax \equiv b \ mod \ n$ 有唯一解，即循环节为 $n-1$ , $b \in [1, n)$ 中的每个值都可以覆盖到
若 $(a, n) \neq 1$ ，设 $g = (a, n)$ ，则 $b$ 会在上述结论上等比 $\times g$ 。故将 $a=\frac{a}{g}, \ n=\frac{n}{g}$  ，在互质的条件下 $b=\frac{b}{g}$  。例如求 $B \leq A$ ，可以先将其 $a, n$ 除以最大公约数 $g$ ，即求新的 $B \leq \frac{A}{g}$ 

对于任意正整数 $X$ ， $X$ 的所有正因子的几何平均数为 $\sqrt{X}$ 

****

## 质数相关

### Legendre 公式

对于质数 $p$ ，函数 $v_p(n)$ 为 $n$ 标准分解后 $p$ 的次数
显然有：$v_p(n!) = \sum_{i=1}^{\infty} \lfloor \frac{n}{p^i} \rfloor$
令函数 $s_p(n)$ 为 $n$ 在 $p$ 进制下的数位和
有：$v_p(n!)=\frac{n-s_p(n)}{p - 1}$

### Kummer 定理

二项式系数 $v_p(\big( ^n_m \big)) = \frac{s_p(m)+s_p(n-m)-s_p(n)}{p-1}$ 
同时也等于在 $p$ 进制下运算 $n-m$ 时退位的次数
多项式系数 $\big( ^{n}_{m_1, \dots , m_k} \big) = \frac{n!}{m_1! \dots m_k!}$
有：$v_p(\big( ^n_{m_1, \dots, m_k} \big)) = \frac{\sum_{i=1}^k s_p(m_i) - s_p(n)}{p-1}$

****

## 位运算

$a + b = (a \& b) \times 2 + (a \oplus b )$ 

$x$ 在第 $i$ 位为 $1$ ，则 $x \% 2^{i+1}$ 在 $[2^i, 2^{i+1})$ ，适用于判断 $x+y$ 在第 $i$ 位是否为 $1$ （满足 $2^i \leq x+y < 2^{i+1}$ 或者 $2^{i+1}+2^i \leq x+y < 2 \times 2^{i+1}$）

$a + b = a | b + a\&b$ 

前缀后缀的 $or$ 和，只会分成 $O(logV)$ 个段

****

## 同余

对于 $a*b \equiv c * d(mod \ m)$ 的解的个数的函数 $f(m)$ 为积性函数，$f(p^e) = p^{3e} + p^{3e-1} -p^{2e-1}$ 

### 皮萨诺周期

在数论中，自然数 $n$ 的皮萨诺周期 $\pi(n)$ 是斐波那契数列模 $n$ 后的周期。$\pi(n) \leq 6n$ 

****

## Gcd and Lcm

若序列 ${a_1, a_2, \dots, a_k}$ 的$\ gcd\ 为 \ x, \ lcm \ 为 \ y$ ，则需满足以下条件：

令 $z = \frac{y}{x}$， 将 $z$ 质因数分解 $z = \Pi_{i=1}^{m}p_i ^{c_i}$ ，将$\frac{a_j}{x}$ 质因数分解 $\frac{a_j}{x} = \Pi_{i=1}^m p_i^{d_i}$ 则每一位质因数对于 $i \in [1, k]$ ，都需要满足 $min\{ d_i\} = 0$ 且 $max \{d_i \} = c_i$ 

区间 $gcd$ 和 $lcm$ 的数量不会超过 $2nlog_2(n)$ ，故求所有子区间 $gcd \ \  lcm \ \ or \ \ and$ 时可以直接固定右端点，对之前的 $lcm$ 与当前枚举的数进行操作并存入集合。

$gcd(x, y) = 1$ 等价于 $\sum_{i|x 且 i|y}\mu(i) = 1$  

对于求形如 $w = \frac{x}{y}$ 的计数问题时，考虑统计 $gcd(x,y) = 1$ 时的方案，先计算出包含 $y$ 的所有方案，再减去 $y$ 的所有因子的方案数

例如，$\{x_i \}$ 为一组序列，求这组序列能形成多少个不同的 $\frac{1}{x_i},\frac{2}{x_i},\dots,\frac{k_i}{x_i}$ ，$dp[x_i] = k_i - \sum_{d | x_i}dp[d], \ ans += dp[x_i]$  

常考虑容斥原理，求 $gcd$ 是 $x$ 的倍数或是 $x$ 的因数的方案数，再按顺序或逆序将多余的方案数减去

****

## Mex

一个序列中，最短的取 $mex$ 的子段（即两端延申的数不影响当前子段的 $mex$ ）最多有 $O(n)$ 个

一个序列中本质不同的 $mex$ 区间最多有 $2n$ 个，即以每个点作为左端点时，不同 $mex$ 的总数量

****

## 摩尔投票

用来解决绝对众数问题：过程非常简单，每个人有一张票，投给第 $x$ 个人，若当前候选人中有第 $x$ 人，则第 $x$ 人的票数加一；若候选人未满或有人得票数为零，则往候选人中添加第 $x$ 个人；否则则将所有候选人的票数减一。

拓展摩尔投票，即要选出 $N$ 个候选人

摩尔投票支持线段树上的合并操作

摩尔投票需要验证正确性（根据具体情况来定）

****

## 原根

对于 $(a, n) \equiv 1$ 的正整数，满足 $a^x \equiv 1(mod\ n)$ 的最小正整数 $x$ ，称之为 $a$ 模 $n$ 的阶，并且 $x | \varphi(n)$  

对于质数 $n$ ，若整数 $a$ 的阶为 $x$ ，则恰好有 $\varphi(x)$ 个互不同余的整数阶为 $x$ 

对于质数 $n$ ，若整数 $a$ 的阶为 $x$ ，则 $a^1,a^2,\dots,a^x$ 模 $n$ 两两不同余

对于质数 $n$ ，若整数 $a$ 的阶为 $x$ ，则 $a^y \equiv 1 \ (mod \ n), x | y$ 

求阶的函数为积性函数

若 $g$ 模 $p$ 的阶恰为 $\varphi(p)$ ，则称 $g$ 为 $p$ 的一个原根

设素数 $p$ 的一个原根是 $g$ ，那么 $g$ 满足对于所有的 $0 \leq i \leq p - 2, \ g^i \ mod\ p$ 各不相同

若一个数 $m$ 有原根，则原根的数量级为 $\varphi(\varphi(m))$ 

若 $g$ 为 $m$ 的最小原根，且 $(i, \varphi(m)) = 1$ ，则 $g^i (mod\ m)$ 也是 $m$ 的原根

### 原根存在定理

只有模 $2,\ 4,\ p^a,\ 2p^a$ 存在原根 （$p$ 是奇质数）

### 原根判定定理

设 $m > 1$，$g$ 为正整数且 $(g, m) = 1$ 。则 $g$ 是 $m$ 的原根当且仅当对于任意 $\varphi(m)$ 的质因子 $q_i$ ，$g^{\frac{\varphi(m)}{q_i}} \not\equiv 1\ (mod \ m)$

****

## 整除分块

$$
下取整分块：r = \lfloor \frac{n}{\frac{n}{l}} \rfloor \\
上取整分块：r = \lfloor \frac{n - 1}{\frac{n - 1}{l}} \rfloor \\
二维数论分块：r = min( \frac{n}{\frac{n}{l}}, \frac{m}{\frac{m}{l}})
$$

****

## 斐波那契数列

$f_{i+k} = f_{k-1}f_i + f_kf_{i+1}$ 



****

## 佩尔方程

$x^2 - dy^2 = 1$

若 $d$ 为平方数，则方程就为 $(x - \sqrt{d}y)(x + \sqrt{d}y)=1$ ，此时只有 $x = \pm 1, \ y = 0$ 这两组特解

若 $d$ 不为平方数，则 $x_n = x_0x_{n - 1} + dy_0y_{n-1} \quad y_n=x_0y_{n-1}+y_0x_{n-1}$ 

****

## 组合数学

$C^0_n + C^2_n + \dots = C^1_n + C_n^3 + \dots = 2 ^{n - 1}$ 

$(^i_j)$ 为奇数当且仅当二进制位上 $j$ 是 $i$ 的子集，可以推广到多重全排列，即所有的 $j$ 组成为 $i$ 的一个二进制划分

$C_a^i + C_{a + 1}^{i} + C_{a + 2}^{i} + \dots + C_{a+n}^{i} = C_{a+n + 1}^{i + 1} - C_a^{i + 1}$ 

$C_{a+i}^i + C_{a + i + 1}^{i + 1} + C_{a + i + 2}^{i + 2} + \dots + C_{a + n}^{n} = C_{a + n + 1}^{n} - C_{a + i}^{i-1}$ 

$0 \times C_n^0 + 1 \times C_n^1 + 2 \times C_n^2 + \dots + n \times C_n^n = n \times 2 ^{n - 1}$ 

$0^2 \times C_n^0 + 1^2 \times C_n^1 + 2^2 \times C_n^2 + \dots + n^2 \times C_n^n = n \times (n + 1) \times 2 ^{n - 2}$ 

$\sum_{i=0}^{n} (C_n^i)^2 = C_{2n}^n$ 

$\sum_{i=0}^k C_n^iC_m^{k-i} = C_{n+m}^{k}$ 

$(C_{n + k}^k)^2 = \sum_{j=0}^{k}(C_k^j)^2C_{n + 2\times k -j}^{2 \times k}$ 

$\sum_{i=1}^{n}i \times (n + 1 - i) = 1 \times n + 2 \times (n - 1) + \dots + n \times 1 = \frac{n\times(n+1)\times(n+2)}{6}$ 常见于含绝对值函数的求和

$j\big( ^{i+j-1}_{i-1} \big) = j\big( ^{i+j-1}_{j} \big) = i\big( ^{i+j-1}_{j - 1} \big) = i\big( ^{i+j-1}_{i} \big)$ 



### 偏序集

设 $S$ 是 $n$ 元素集合

$S$ 的一个反链是集合 $S$ 的一个子集的一个集合 $\mathcal{A}$ ，其中 $\mathcal{A}$ 中的子集不相互包含

$S$ 的一条链是集合 $S$ 的一个子集的一个集合 $\mathcal{C}$ ，对于 $\mathcal{C}$ 中的每一对子集，总有一个包含在另一个之中



设 $(X, \leq)$ 是有限偏序集

$X$ 的反链是 $X$ 的一个子集 $\mathcal{A}$ ，它的任意两个元素都不可比

$X$ 的链是 $X$ 的一个子集 $\mathcal{C}$ ，它的每一对元素都可比

设 $r$ 是链的最大大小，则 $X$ 可以被划分成 $r$ 个反链，但不能划分成少于 $r$ 个反链

设 $m$ 是反链的最大大小，则 $X$ 可以被划分为 $m$ 个链，但不能划分成少于 $m$ 个链



### 卡特兰数

在前 $2n$ 项的前缀和中，往左走的个数总大于等于往上走的个数，由方格图证明，沿对角线折叠构造

$$
f(n) = \big( _{n} ^{2n} \big) - \big( _{n - 1} ^{2n} \big) \\递推式满足：f(n) = f(1) \times f(n - 1) + f(2) \times f(n - 2) \dots
$$



设 $h_n$ 表示把凸多边形区域分成三角形区域的方法数：在有 $n + 1$ 条边的凸多边形区域内通过插入在其中不相交的对角线而把它分成三角形区域。定义 $h_1=1$ ，则 $h_n = h_1h_{n-1} + h_2h_{n-2}+\dots+h_{n-1}h_{1}=\sum_{i=1}^{n-1}h_ih_{n-i}(n\geq2)$  ，这个递推关系的解是 $h_n=\frac{1}{n}\big(^{2n-2}_{n-1}\big)$  

$n$ 个点能组成的二叉树的个数

#### Raney 引理

对于 $n$ 个整数 $x_1, x_2, \dots x_n$ ，定义 $sum_i = \sum_{j=1}^i x_i$ ，若有 $sum_n=1$ ，则其所有循环位移中有且仅有一个满足 $\forall i, sum_i > 0$ 
可推导卡特兰数

### 贝尔数

基数为 $n$ 的集合划分的方案数，记作 $B_n$

$$
B_n = \sum_{k = 0}^{n - 1}(_{k}^{n - 1})B_k \\B_0 = 1
$$



### 斯特林数

#### 第一类斯特林数

将 $1 \dots n$ 划分成 $k$ 个非空圆排列的方案数，记作 $s(n, k)$ 或 $[_{k}^{n}]$

##### 性质

$[_{k}^{n}] = [_{k - 1}^{n - 1}] + (n - 1) \times [_{k}^{n - 1}]$ 

$s(0, 0) = 1 \ s(n, 0) = 0$ 

$s(n, 2) = (n - 1)! \times \sum_{i = 1}^{n - 1} \frac{1}{i}$ 

$s(n, n - 1) = C(n, 2)$ 

$s(n, n - 2) = 2 \times C(n, 3) + 3 \times C(n, 4)$ 

$\sum_{k = 0}^{n} [_{k}^{n}] = n!$ 

$x^{n \uparrow} = \sum_{k = 0}^{n} [_{k}^{n}] x^{k} \Rightarrow 用于求某一行的生成函数$ 

$\sum_{n\geq0}(-1)^{n + k}[^n_k] \frac{x^n}{n!} = \frac{1}{k!}(ln(1 + x))^k \Rightarrow 用于求某一列的生成函数$ 

#### 第二类斯特林数

将 $1 \dots n$ 划分成 $k$ 个非空子集的方案数，记作 $S(n, k)$ 或 $\{_{k}^{n} \}$

$n$ 个有标号的球分配到 $k$ 个无标号的盒子的方案数，每个盒子至少一个球

##### 性质

$\{ _{k}^{n} \}\%2=1$ 当且仅当：令 $t = \lfloor \frac{k - 1}{2} \rfloor$ , $t \& (n - k) = 0$ 

$\{ _{k}^{n} \} = \{ _{k - 1}^{n - 1} \} + k \times \{ _{k}^{n - 1} \}$ 

$S(n, 0) = 0^n \ S(n, 1) = 1 \ S(n, n) = 1$ 

$S(n, 2) = 2^{n - 1} - 1$ 

$S(n, n - 1) = C(n, 2)$ 

$S(n, n - 2) = C(n, 3) + 3 \times C(n, 4)$ 

$C(n, n - 3) = C(n, 4) + 10 \times C(n, 5) + 15 \times C(n, 6)$ 

$\sum_{k = 0}^{n}S(n, k) = B_n$ 

$S(n, k) = \frac{1}{k!} \times \sum_{i = 0}^{k}(-1)^{i}C(k, i)(k - i)^{n} \Rightarrow 用于求某一行的生成函数$ 

$\sum_{n \geq 0} \frac{\{^n_k\}}{n!}x^n = \frac{1}{k!}(e^x-1)^k \Rightarrow 用于求某一列的生成函数$ 

$n^m = \sum_{k = 0}^{m} \{ ^m _k\} n^{k \downarrow}$ 

#### 两类斯特林数之间的转化关系

$\sum_{k = 0}^{n}S(n, k)s(k, m) = \sum_{k = 0}^{n}s(n, k)S(k, m)$  
类似于矩阵的对称转置关系



### 整数拆分

$n$ 个无标号的球分配到 $k$ 个无标号的盒子的方案数，每个盒子至少一个球，记为 $p(n, k)$ 

$递推关系：p(n, k) = p(n - 1, k - 1) + p(n - k, k)$ 

$关于n的常生成函数：\sum_{n \geq 0}p(n,k)x^n = x^k\Pi_{i = 1}^{k}\frac{1}{1 - x^i}$

$n$ 个无标号的球分配到一些无标号盒子的方案数，每个盒子至少一个球，记为 $p(n)$

可知 $p(n) = \sum_{k = 1}^{n}p(n, k)$

$递推关系：p(n) = \sum_{k\geq1}(-1)^{k-1}(p(n - \frac{3k^2-k}{2}) + p(n - \frac{3k^2 + k}{2}))$

$常生成函数：\sum_{n \geq 0}p(n)x^n = \Pi_{i\geq1}\frac{1}{1 - x^i}$ 

### 分配问题

| $n$ 个球 | $k$ 个盒子 | 盒子可以为空                 | 每个盒子至少一个球        |
|:------:|:-------:|:----------------------:|:----------------:|
| 有标号    | 有标号     | $k^n$                  | $k!\{^n_k\}$     |
| 有标号    | 无标号     | $\sum_{i=1}^k\{^n_i\}$ | $\{^n_k\}$       |
| 无标号    | 有标号     | $(^{n + k - 1}_{k-1})$ | $(^{n-1}_{k-1})$ |
| 无标号    | 无标号     | $p(n + k, k)$          | $p(n,k)$         |

### 分配问题（加强版）

把 $n$ 个球放入 $k$ 个盒子，装有 $i$ 个球的盒子有 $f_i$ 种形态 $(f_0 = 0)$ ，不同形态算不同方案，有多少种方案？

设 $\{f_i\}_{i\ge1}$ 的常生成函数为 $F(x) = \sum_{i\geq1}f_ix^i$ ，指数生成函数为 $E(x) = \sum_{i\geq1}f_i\frac{x^i}{i!}$ 

| $n$ 个球 | $k$ 个盒子 | 方案的生成函数                    |
|:------:|:-------:|:--------------------------:|
| 有标号    | 有标号     | $EGF = E(x)^k$             |
| 有标号    | 无标号     | $EGF = \frac{1}{k!}E(x)^k$ |
| 无标号    | 有标号     | $OGF = F(x)^k$             |
| 无标号    | 无标号     |                            |

### 分配问题（加强版 $2$）

把 $n$ 个球放入一些盒子（不限数量），装有 $i$ 个球的盒子有 $f_i$ 种形态 $(f_0 = 0)$ ，不同形态算不同方案，有多少种方案？

设 $\{f_i\}_{i\ge1}$ 的常生成函数为 $F(x) = \sum_{i\geq1}f_ix^i$ ，指数生成函数为 $E(x) = \sum_{i\geq1}f_i\frac{x^i}{i!}$ 

| $n$ 个球 | 盒子  | 方案的生成函数                                                                           |
|:------:|:---:|:---------------------------------------------------------------------------------:|
| 有标号    | 有标号 | $EGF = \frac{1}{1-E(x)}$                                                          |
| 有标号    | 无标号 | $EGF = e^{E(x)}$                                                                  |
| 无标号    | 有标号 | $OGF = \frac{1}{1 - F(x)}$                                                        |
| 无标号    | 无标号 | $OGF = \Pi_{i\geq1}(\frac{1}{1-x^i})^{f_i} = e^{\sum_{j\geq1} \frac{1}{j}F(x^j)}$ |



### 圆排列

$$
(n - 1)!
$$

### 错排问题

$$
D(1) = 0, \  D(2) = 1, \  D(n) = (n - 1) \times [D(n - 1) + D(n - 2)]
$$

### 多重集合的排列

多重全排列是指对于 $k$ 个不同的数，求有 $n_1$ 个 $1$， $n_2$ 个 $2$， $\dots$ ，$n_k$ 个 $k$ 的排列数
设 $n = n_1 + n_2 + \dots + n_k$
$\big( _{n_1, n_2, \dots, n_k} ^{n} \big) =  \frac{n!}{n_1! \times n_2! \times \dots \times n_k!}$

把 $n$ 对象集合划分成 $k$ 个标有标签的盒子，且第 $1$ 个盒子含有 $n_1$ 个对象，第 $2$ 个盒子含有 $n_2$ 个对象，$\dots$ ，第 $k$ 个盒子含有 $n_k$ 个对象，这样的划分方案数等于 $\frac{n!}{n_1!n_2!\dots n_k!}$ 

### 多重集合的组合

如果 $S$ 是多重集合，那么 $S$ 的 $r$ 组合是 $S$ 中的 $r$ 个对象的无序选择

设 $S$ 是有 $k$ 种类型对象的多重集合，每种元素均具有无限的重复数。那么 $S$ 的 $r$ 组合的个数等于 $\big( ^{r - 1 + k} _{ k - 1}\big)$ 

与 $x_1 + x_2 + \dots + x_k = r$ 的非负整数解集合之间存在一一对应的关系

### 康托展开

$$
\begin{aligned}
X = a_{i} \times (n - 1)! + a_{i - 1} \times (n - 2)! + \dots + a_{1} \times 0! \\
其中a_{i}表示原数的第i位在当前未出现的元素中排在第几个\\
\end{aligned}
$$

### 五边形数

五边形数是能排成五边形的多边形数。其概念类似三角形数及平方数，但所对应的形状没有旋转对称的特性

$p_n = \frac{3n^2-n}{2}$ 

利用以下公式可以测试一个正整数 $x$ 是否为五边形数 $n=\frac{\sqrt{24x+1} + 1}{6}$ ，若 $n$ 为自然数，则为五边形数

****

## 欧拉定理

### 降幂

$$
a^b \equiv
\begin{cases}
a^{b\ mod\  \phi(p)} , \ \ \ \ \ \ \ \ \ \ \ \ \ \ gcd(a, p) = 1\\
a^b,  \ \ \ \ \ \ \ \ \ \ \ \ \ \ \  \ \ \ \ \ \ \ \ \ \ \  gcd(a, p) \neq 1 \  \ b < \phi(p)        \ \ \  \ \mod(p) \\
a^{b \ mod \ \phi(p) \ + \phi(p)}, \ \ \ \ \ \ gcd(a, p) \neq 1 \ \  b \geq \phi(p)
\end{cases}
$$

若求最小的的正整数满足 $a^x \equiv 1(\ \% \ c)$，则 $x$ 一定是 $phi(c)$ 的一个约数

即若最外层 $\% \ p$  ，则一层幂上 $\% \ \varphi(p)$ ，二层幂上 $\% \ \varphi(\varphi(p))$ $\dots$ 

****

## 容斥原理

常考虑每个位置上满足某个性质的方案数，对于其他位置可以任意取

对求 $f(val = x)$ 的情况，可以考虑求 $f(val \leq x) - f(val \leq x - 1)$ 

设 $A_i = {x:x属于S且x具有性质P_i} \ (i = 1,2,\dots,m)$ 

集合 $S$ 中不具有性质 $P_1,P_2,\dots,P_m$ 的对象个数：

$|\bar{A_1} \cap \bar{A_2} \cap \dots \cap \bar{A_m}| = |S| - \sum|A_i| + \sum|A_i \cap A_j| - \dots$ 

集合 $S$ 中至少具有性质 $P_1,P_2,\dots ,P_m$ 之一的对象个数：

$|{A_1} \cup {A_2} \cup \dots \cup {A_m}| = \sum|A_i| - \sum|A_i \cap A_j| + \dots$ 

### 一些tricks

* 例1

$$
x_1 + \dots + x_k \leq n \Leftrightarrow x_1 + \dots + x_k + z = n \quad 其中z \geq 0
$$

* 例2

求 $x_1 + \dots + x_k = n 且 l_i \leq x_i \leq r_i$ 的解 
容斥原理: $S = \{x_1 + \dots + x_k = n 且 \forall x_i \geq l_i\}, \quad A_i = \{x_1 + \dots + x_k = n 且 \forall x_i \geq r_i + 1\}$
$ans = S - \sum_{i \in k} \bigcup A_i$ 

长度为 $n$ 的 $01$ 串，其中 $m$ 个为 $1$ ，求最长连续 $1$ 恰好为 $k$ 的方案数：
首先容斥，求最长连续 $1 \leq k$  的方案数，考虑在 $0$ 的间隙中插入 $1$ ，有 $n-m$ 个 $0$ 故有 $n-m+1$ 个空隙。再次考虑容斥，求至少有 $i \leq n-m+1$ 个位置存在 $> k$ 的连续 $1$ 段， 即总和 $\sum_{i=1}^{n-m+1}x_i = m-i*(k+1), x_i \geq 0$  $\Rightarrow$ $\big(^{n-i*(k+1)} _{n-m} \big)$ 



### 二项式反演

记 $f(n)$ 表示恰好使用 $n$ 个不同元素形成特定结构的方案数，$g(n)$ 表示从 $n$ 个不同元素中选出 $i \geq 0$ 个元素形成特定结构的总方案数

若已知 $f(n)$ 求 $g(n)$ ，那么显然有：$g(n)=\sum_{i=n}^n\big( ^n _i \big) f(i)$ 

若已知 $g(n)$ 求 $f(n)$ ，那么：$f(n) = \sum_{i=0}^n \big( ^n_i \big)(-1)^{n-i} g(i)$ 

### 子集反演

现在有满足某种条件的元素集合 $A$ 

设 $f(S)$ 表示 $S = A$ 的答案，$g(S)$ 表示 $S \subseteq A$ 的答案，我们枚举钦定了某个集合 $T$ ，则有 $g(S) = \sum_{T \subseteq S}f(T)$ ，此时子集反演给出 $f(S) = \sum_{T \subseteq S}(-1)^{|S| - |T|}g(T)$  

设 $f(S)$ 表示 $S = A$ 的答案，$g(S)$ 表示 $S \supseteq A$ 的答案，我们枚举钦定了某个集合 $T$ ，则有 $g(S) = \sum_{T \supseteq S}f(T)$ ，此时子集反演给出 $f(S) = \sum_{T \supseteq S}(-1)^{|T| - |S|}g(T)$  

例如：假如题目给出 $n$ 个元素（或是转换为 $n$ 个条件等形式），要求 $f(S)$ 表示恰好选择 $S$ 中元素的方案数，参照前面的思想，可以记 $g(S)$ 表示至多选 $S$ 中这些点的方案数（这样 $S$ 中的点可以被多次选择），即：“恰好是这个集合” $\Rightarrow$ “至多是这个集合”

****

## 狄利克雷卷积

$$
(f \star g)(n) = \sum_{d | n} f(d) \times g(\frac{n}{d})
$$



### 莫比乌斯反演

$f(n) = \sum_{d | n} g(d) \Leftrightarrow g(n) = \sum_{d | n}f(d) \times \mu(\frac{n}{d})$ 

$f(n) = \sum_{n | m \ m \leq N}g(m) \Leftrightarrow g(n) = \sum_{n | m \ m \leq N} f(m) \times \mu(\frac{m}{n})$

#### 无平方因子数

$1$ 到 $n$ 中无平方因子数的个数是：

$Q(n) = \sum_{k \leq n} \mu^2(k) =  \sum_{k \leq \sqrt{n}} \mu(k) \ [ \frac{n}{k^2} ]$

#### 环计数问题

假设 $n$ 个元组成一个环，每个元都是 $1,2,\dots,r$ 中的一个数，两个环是不同的环当且仅当它们不能通过旋转使得两个环中对应的每个元素都相等。求有多少个这样的环

$solution：$ 对于序列而言，共有 $r^n$ 种方案，将环展开并无限延申，考虑每个周期为 $d$ 的序列，易得 $d$ 是 $n$ 的一个因子，设 $f(d)$ 为周期为 $d$ 的序列的数目，得出 $r^n = \sum_{d|n}df(d)$，答案即为 $\sum_{d|n}f(d)$

### 一些 $tricks$

* 例1
  求 $\sum_{1 \leq x \leq n \  1 \leq y \leq n} [a_{b_{x}} = b_{a_{y}}] \times [gcd(x, y) = 1]$ 

设 $f(d) = \sum_{1 \leq x \leq n \  1 \leq y \leq n} [a_{b_{x}} = b_{a_{y}}] \times [gcd(x, y) = d]$ 

令 $g(d) = \sum_{d | d'}f(d')$ 
$\therefore f(d) = \sum_{d | d'}g(d') \times \mu(\frac{d'}{d})$ 
所求即 $f(1) = \sum_{d = 1}^{n}g(d) \times \mu(d)$

* 例2

$$
\sum_{i = 1} ^ {n} \sum_{j = 1} ^ {m}f(gcd(i, j)) \Rightarrow \sum_{d = 1} ^ {min(n, m)} g(d) \times \lfloor \frac{n} {d} \rfloor \times \lfloor \frac{m} {d} \rfloor \\
其中f(n) = \sum_{d | n} g(d)\\
即g = f \star \mu
$$

常见求：

 $\sum_{i = 1} ^ {n} \sum_{j = 1} ^ {m}gcd(i, j) = \sum_{d=1} \varphi(d) \times \lfloor \frac{n} {d} \rfloor \times \lfloor \frac{m} {d} \rfloor$ 利用 $\mu \star id = \varphi$

 $\sum_{i = 1} ^ {n} \sum_{j = 1} ^ {m} [gcd(i, j)=1] = \sum_{d=1} \mu(d) \times \lfloor \frac{n} {d} \rfloor \times \lfloor \frac{m} {d} \rfloor$ 利用 $\mu\star \epsilon = \mu$ 

对于在计数问题，若$val(x)$ 与 $val(y)$ 在 $gcd(x, y)$ 上有重复计数，则从小到大枚举每一个 $i$ 时，将所有 $i$ 的倍数全部减去当前 $val(i)$ ，使重复部分只在 $gcd$ 处计算

****

## 常见积性函数及结论

若 $f(n)$ 为积性函数，则 $g(n) = \sum_{d | n} f(d)$ ，$nf(n)$ ，$\frac{f(n)}{n}$也为积性函数 

$元函数\ \epsilon(n) = [n = 1]$
$常数函数\ 1(n) = 1$
$恒等函数\ id(n) = n$

$\mu \star 1 = \epsilon$
$\mu \star id = \varphi$
$\varphi \star 1 = id$
$f \star \epsilon = f$
$f \star 1 \neq f$

假设欧拉函数的前缀和为 $s_k = \sum_{i = 1}^{k} \varphi(i)$ ，则 $\sum_{i=1}^{n}\sum_{j=1}^{n}[gcd(i, j) =1] = 2s_n-1$ 

$\sum_{i=1}^{n}\sum_{j=1}^{i} \lfloor \frac{i}{j} \rfloor = \sum_{i+j=n+1} i \times d(j)$ 其中，$d(j)$ 表示 $j$ 的约数个数 



### $Jordan's\ totient\ function$

$J_k(n)$ 表示从小于等于 $n$ 中选取 $k$ 个数，与 $n$ 一同构成大小为 $k+1$ 的互素的集合数量，其中 $\varphi(n)$ 是特殊的 $J_k(n)$ ，即 $J_1(n)$ 

$J_k(n) = n^k\Pi_{p|n, \ p\ is\ prime} (1-\frac{1}{p^k})$  

$\sum_{d|n} J_k(d)=n^k$ 

$J_k(n) = \mu(n) \star n^k$ 

$|GL(m,Z/n)| = n^{\frac{m(m-1)}{2}}\Pi_{k=1}^mJ_k(n)$ 

$|SL(m,Z/n)| = n^{\frac{m(m-1)}{2}}\Pi_{k=2}^mJ_k(n)$ 

$J_2(n):1,3,8,12,24,24,48,48,72,72,120,96,168,144,192,\dots$ 



### $n*\sigma(n)$

设 $f = n * \sigma(n)$ ，$g = f\star \mu$ 

$f = 1,6,12,28,30,72,56,120,117,180,132,336,182,336,\dots$ 

$f(p^e) = \frac{p^e * (p^{e+1}-1)}{p-1}$ 

$g = 1,5,11,22,29,55,55,92,105,145,131,242, 181,275,319\dots$ 

$g(p^e) = p^{e-1} * (p^e*(p+1)-1)$ 

$h(p^e) = p^e*g(p^e)=p^{3e} + p^{3e-1}-p^{2e-1}$ ，即 $|SL(2,Z/p^e)|$ 

****

## 生成函数

$对于数列\{1,\ 1,\ \dots , \ 1 \}$

$常生成函数 A(x) = 1 + x + x^2 + \dots + x^n + \dots = \frac{1}{1 - x}$

$指数生成函数 A(x) = 1 + x + \frac{x^2}{2!} + \dots + \frac{x^n}{n!} + \dots = e^x$

$A(x) = 1 + ax + a^2x^2 + \dots + a^nx^n + \dots = \frac{1}{1 - ax}$

$A(x) = \big( ^{k -1} _{0} \big) + \big( ^k _1\big) x + \big( ^{k + 1}_{2}\big)x^2 + \dots = \sum_{i \geq 0} \big( ^{i + k - 1} _{i}\big)x^i = \frac{1}{(1 - x)^k}$

$A(x) = \big( ^{k -1} _{0} \big) + \big( ^k _1\big) ax + \big( ^{k + 1}_{2}\big)a^2x^2 + \dots = \sum_{i \geq 0} \big( ^{i + k - 1} _{i}\big)a^ix^i = \frac{1}{(1 - ax)^k}$

$A(x) = 1 + x + \dots + x^n = \frac{1 - x^{n + 1}}{1 - x}$

$A(x) = 1 + ax + a^2\frac{x^2}{2!} + \dots = \sum_{n \geq 0}a^n \frac{x^n}{n!} = e^{ax}$

$A(x) = 1 + \frac{x^2}{2!} + \frac{x^4}{4!} + \dots = \frac{e^x + e^{-x}}{2}$

$A(x) = x + \frac{x^3}{3!} + \frac{x^5}{5!} + \dots = \frac{e^x - e^{-x}}{2}$

$A(x) = x - \frac{x^2}{2} + \frac{x^3}{3} - \frac{x^4}{4} + \dots + (-1)^{n + 1}\frac{x^n}{n} + \dots = ln(1+x)$

$x + 2^2x^2 + 3^2x^3+\dots+n^2x^n+ \dots = \sum_{i=1}^{\infty}i^2x^i = \frac{x(1+x)}{(1-x)^3}$



### 牛顿二项式

$$
\big( _m ^n \big) = \frac{n \times (n - 1) \dots \times (n - m + 1)}{m!} \ 对于任何实数\ n\ 成立
$$

> 例如：

$$
(^{\frac{1}{2}}_n) = \frac{(\frac{1}{2}) \times (\frac{1}{2} - 1) \dots \times (\frac{1}{2} - n + 1)}{n!}
$$

$$
(1 + ax)^n = \sum_{i \geq 0}\big( ^n_i \big)a^i x^i \ 对于任何实数\ n\ 成立
$$

### 常生成函数

$设\ S = \{ a_1, a_2, \dots, a_k \}，且a_i可以取的次数的集合为M_i，记F_i(x) = \sum_{u \in Mi}x^u$

$则从\ S\ 中取\ n\ 个元素组成集合的方案数\ g(n)\ 的常生成函数\ G(x) = \sum_{i \geq 0}g(i)x^i，满足$
$G(x) = F_1(x)F_2(x)\dots F_k(x)$

> 有两种物体，其中取 $i$ 个第一种物体的方案数为$a_i$ ，取 $j$ 个第二种物体的方案数为 $b_j$ ，求取 $k$ 个物体的方案数。



设 $n$ 为正整数，$1(1 + x)(1+x+x^2)(1+x+x^2+x^3)\dots(1+x+x^2+\dots+x^{n-1})=\frac{\Pi_{j=1}^n(1-x^j)}{(1-x)^n}$ 



### 指数生成函数

$设\ S = \{ a_1, a_2, \dots, a_k \}，且a_i可以取的次数的集合为M_i，记F_i(x) = \sum_{u \in Mi}\frac{x^u}{u!}$
$则从\ S\ 中取\ n\ 个元素排成一列的方案数\ g(n)\ 的指数生成函数\ G(x) = \sum_{i \geq 0}g(i)\frac{x^i}{i!}，满足 G(x) = F_1(x)F_2(x)\dots F_k(x)$ 

> 有两种物体，其中取 $i$ 个第 $1$ 种物品的方案数为 $a_i$，取 $j$ 个第 $2$ 种物品的方案数为 $b_j$，求取 $k$ 个物品并排成一列的方案数。

> 用数字 $1,2,3,4$ 作 $6$ 位数，每个数字在六位数种出现的次数不得大于 $2$，求出可以作出多少个不同的 $6$ 位数。



设 $S$ 是多重集合 $\{n_1·a_1，n_2·a_2 ，\dots，n_k·a_k\}$ ，其中 $n_1，n_2，\dots ，n_k$ 是非负整数。设 $h_n$ 是 $S$ 的 $n$ 排列数。那么数列 $h_0，h_1，h_2，\dots，h_n，\dots$ 的指数生成函数 $g^{(e)}(x) = f_{n_1}(x)f_{n_2}(x) \dots f_{n_k}(x)$  ，其中，对于 $i=1,2,\dots,k,$ 有 $f_{n_i}(x) = 1 + x + \frac{x^2}{2!} + \dots + \frac{x^{n_i}}{n_i}$ 

可重集可以意味着有序

转化思想：对于排列，不要只思考将每个数放在哪个位置，还要去想可以把哪个下标放在哪个数上

### 欧拉数

$\big< ^n _k \big>$ 表示长度为 $n$ 且有 $k$ 个位置升高的排列的个数

$\big< ^n _k \big> = (k + 1)\big< ^{n-1} _k \big> + (n - k)\big< ^{n-1} _{k-1} \big>$ 

$\big< ^n _m \big> = \sum_{k=0}^m \big(^{n +1}_{k} \big) (-1)^{k}(m + 1-k)^n$ 

### 单位根反演

取模意义下的单位根的值 $w_1 = g^{\frac{P-1}{n}}$ ，其中必须要求 $n|(P-1)$ ，$g$ 为 $P$ 的原根

单位根反演的核心式子为：$[n|k]=\frac{1}{n} \sum_{i=0}^{n-1}w_n^{ik}$ ，如果 $n\%k=0$ ，右式为 $1$ ，否则为 $0$ 

推论：设有生成函数  为 $m$ 次多项式，其在 $x^k$ 处的系数为 $f_k$ ，那么有 $\sum_{k=0}^m[n|k]f_k=\frac{1}{n} \sum_{i=0}^{n-1}f(w_n^i)$ 

****

## 多项式计数

多项式的指数：
对于形式幂级数 $f(x)$ ，若 $f(0)=0$ （即常数项 $a_0=0$），则指数函数定义为：
$e^{f(x)} = \sum_{n=0}\frac{f(x)^n}{n!} = 1 + f(x) + \frac{f(x)^2}{2!} + \frac{f(x)^3}{3!}+\dots$ 

多项式的对数：
对于形式幂级数 $f(x)$ ，若 $f(0)=1$ （即常数项 $a_0=1$），则对数函数定义为：
$log(f(x)) = \sum_{n=1}(-1)^{n+1}\frac{(f(x)-1)^n}{n} = \frac{f(x)-1}{1} + \frac{(f(x)-1)^2}{2} + \frac{(f(x)-1)^3}{3}+\dots$ 
常用 $log(f(x)+1)$ 

多项式取模：
多项式取模运算 $f(x) \% g(x)$ 得到的是满足以下条件的多项式 $r(x)$ ：

1. 存在多项式 $q(x)$ 使得 $f(x) =q(x) \times g(x) + r(x)$ 
2. $r(x)$ 的次数严格小于 $g(x)$ 的次数

$k$ 阶线性递推数列是指满足以下递推关系的数列 $\{ a_n \}$ ：
$a_n =c_1a_{n-1} + c_2a_{n-2}+\dots +c_ka_{n-k}$ 
其中 $c_1, c_2, \dots, c_k$ 是常数且 $c_k \neq 0$ ，对应的特征多项式定义为：$P(x) = x^k-c_1k^{k-1}-c_2k^{k-2} - \dots - c_k$ 
线性递推数列的计算可以转化为多项式取模问题。具体来说，计算递推数列的第 $n$ 项等价于计算 $x^n \ \% P(x)$ 的结果

### 二项式相关

#### 前缀和与差分

$F(x) = \sum_{i=0}^{\infty}a_ix^i$

$F(x)$ 的 $k$ 维前缀和为 $F \times \frac{1}{(1-x)^k}$ 或 $F \times \sum_{i=0}^{\infty}(^{k-1+i}_{i})x^i$ 或 $\sum^kf(x) = \sum_{i=0} (^{k + i - 1} _{i})f(x - i)$

$F(x)$ 的 $k$ 维差分为 $F \times (1-x)^k$ 或 $F \times \sum_{i=0}^{\infty}(-1)^i(^k_i)x^i$ 或 $\Delta ^n f(x) = \sum_{i=0}^k(^k_i)(-1)^{k-i}f(x+i)$ 

$k$ 维前缀和卷积式子 $b[i] = b[i - 1]*(k+i-1)/i$

$k$ 维差分卷积式子 $b[i] = -b[i - 1]*(k-i+1)/i$ 



### 半在线卷积

1. 分治做法，每层再进行一次多项式乘法，将左半部分的影响加和到右半部分
2. 推式子，多项式求逆

****

## 拉格朗日插值

$n$ 个点值 $(x_i, y_i),(1 \leq i \leq n)$ ，满足 $x_i \neq x_j$ ,他们唯一确定一个 $n - 1$ 次多项式 $f(x)$ ：
$$
f(x) = \sum_{i = 1}^{n}y_i\prod_{j \neq i}\frac{x - x_j}{x_i - x_j}
$$

### 特殊情况

只需求一个 $f(x_0)$ , 且 $x_i = i$ , 则可以 $O(n)$ 求出



若 $deg\ f_0 = k$ ，$f_m(n) = \sum{f_{m-1}(i)}$ ，则 $f_m(n)$ 是一个次数为 $m + k$ 的多项式

****

## 牛顿迭代

给定多项式 $g(x)$，求满足 $g(f(x)) = 0$ 的形式幂级数 $f(x)$

$n = 1时，解\ g(a_0) = 0$

$假设已经求出了前\ n\ 项\ f(x) \equiv f_0(x) = a_0 + a_1x + \dots + a_{n - 1}x^{n - 1}(mod \ x^n)$ 

$则\ f(x) \equiv f_0(x) - \frac{g(f_0(x))}{g'(f_0(x))}(mod\ x^{2n})$

### 求逆

设 $h(x)$ 时给定的形式幂级数，求它的逆 $f(x)$，则 $g(f(x)) = \frac{1}{f(x)} - h(x) = 0$，得到的迭代式为： $f(x) \equiv 2f_0(x) - f_0^2(x)h(x) \ (mod \ x^{2n})$

$note：$ 当引入 $h(x)$ 后，求导时需求偏导，即无需对 $h(x)$ 求导

### 开方

设 $h(x)$ 时给定的形式幂级数，求它的开方 $f(x)$，则 $g(f(x)) = f(x)^2 - h(x) = 0$，得到的迭代式为：$f(x) \equiv f_0(x) - \frac{f_0^2 - h(x)}{2f_0(x)} \ (mod \ x^{2n})$

****

## 形式幂级数 vs 幂级数

### 简介

形式幂级数的本质是序列，幂级数的本质是极限。即形式幂级数关注的是各项系数，而幂级数关注的是多项式在阶趋近于无穷大时的形式

形式幂级数通过“代入”还原成幂级数

假设系数在 $\mathbb{C}$ 上，可以证明形式幂级数与具有正收敛半径的幂级数在“通常”的所有运算上服从相同的规律

### 形式幂级数的更多运算

假设 $f(x) = a_0 + a_1x + a_2x^2 + \dots + a_nx^n + \dots$
求导：$f'(x) = a_1 + 2a_2x + \dots + (n + 1)a_{n + 1}x^n + \dots$
积分：$\int f(x)dx = a_0x + \frac{a_1}{2}x^2 + \dots + \frac{a_{n - 1}}{n}x^n + \dots$
复合：$f(x) = a_1x + \dots + a_nx^n + \dots， \ g(x) = b_0 + b_1x + \dots + b_nx^n + \dots$ 则 $g$ 复合 $f$ 定义为 $c_0 + c_1x + \dots + c_nx^n + \dots$ ，满足 $c_0 = b_0，c_n = \sum_{k = 1}^{n}b_k\sum_{i_1 + i_2 + \dots + i_k = n}a_{i_1}a_{i_2}\dots a_{i_k}$ ，记作 $g(f(x))$ 或 $g \circ f$ 

复合求导的链式法则：$(g \circ f)'= (g'\circ f) \times f'$

### 指数形式幂级数 vs 对数形式幂级数

$e^x = \sum_{n \geq 0} \frac{1}{n!}x^n$ 

$ln(1 + x) = \sum_{n \geq 1} \frac{(-1)^{n + 1}}{n}x^n$ 

设形式幂级数 $f(x)$ 满足 $[x^0]f(x) = 0$ ，由此可定义 $e^{f(x)}$ 与 $ln(1 + f(x))$ 

$g(x) = e^{f(x)} \Leftrightarrow f(x) = ln(g(x))$ 

****

## 多项式常见应用

$01$ 背包：分治 $FFT$

无穷背包计数：利用 $ln$ 将多项式相乘转为加法，再求 $exp$ ，注意到等式 $ln(1-x^V) = -\sum_{i \ge 1} \frac{x^{V \times i}}{i}$ ，通过两边求导得到

****

## 集合幂级数

设 $f_i$  为 $i$ 的超集和，即 $f_i$ 统计的是满足条件的元素的数量，而 $F_i$ 为 $\sum_{i \& i' = i}f_{i'}$ ，即 $F_i$ 统计的满足条件的集合的数量，故称集合幂级数

****

## 群论

解决考虑旋转或是翻转计数问题的理论

### Burnside 引理

设有限群 $(G,\circ)$ 作用在有限集 $X$ 上， 则 $X$ 上的 $G-$ 轨道数量为： $N = \frac{1}{|G|}\sum_{g \in G}\psi(g)$

其中 $\psi(g)$ 表示 $g(x) = x$ 的 $x$ 的数量（即求群中置换环的数量）

> 例：给一个六元环的节点涂上颜色，有 $m$ 种颜色，通过旋转可以得到的方案算同一个方案，求方案数
> 
> $solution：$ 令 $X = \{[a_1, \dots,a_6],1\leq a_i \leq m\}$ ，$G$ 是 $6$ 个置换组成的旋转群，分别是：
> 
> $\bullet e=[1,2,3,4,5,6],\ \psi(e) = m^6$ 
> 
> $\bullet g_1=[2,3,4,5,6,1],\ \psi(g_1) = m$
> 
> $\bullet g_2 =[3,4,5,6,1,2],\ \psi(g_2) = m^2$
> 
> $\bullet g_3=[4,5,6,1,2,3],\ \psi(g_3) = m^3$
> 
> $\bullet g_4=[5,6,1,2,3,4],\ \psi(g_4)=m^2$
> 
> $\bullet g_5=[6,1,2,3,4,5],\ \psi(g_5) = m$
> 
> 因此 $N = \frac{1}{6}(m^6+m^3+2m^2+2m)$

染色问题：

给一个 $n$ 元环的节点涂上颜色，有 $m$ 种颜色，通过旋转可以得到的方案算同一种方案，求方案数

$\bullet$ 令 $X = \{x|x:\{ a_1,\dots,a_n\} \rightarrow \{ b_1,\dots,b_m\}\}$ ，$G$ 是正 $n$ 边形的旋转群，其第 $i(0\leq i < n)$ 个元素是 $g_i = [i + 1, i+2,\dots,n,1,\dots,i]$ 

$\bullet \ \psi(g_i) = m^{(n,i)}$ 

$\bullet \ N = \frac{1}{n}\sum_{i=0}^{n - 1}m^{(n,i)} = \frac{1}{n}\sum_{d|n}\varphi(d)m^{n/d}$ 

拓展来说：对长度为 $n$ 的轮换群，若一个旋转使序列分为 $g$ 个循环，等价于周期 $g$ 的块重复 $t=\frac{n}{g}$ 次。即在上面式子中的 $m^{\frac{n}{d}}$ ，在一个长度为 $n$ 的轮转群中重复了 $d$ 次$(\frac{n}{d} \times d = n)$，并对答案计数产生了 $\varphi(d)$ 的贡献。

#### 置换群的轮换指标

设 $(G,\circ)$ 是一个 $n$ 元置换的置换群，它的轮换指标为 $P_G(x_1,x_2,\dots,x_n) = \frac{1}{|G|}\sum_{g \in G}x_1^{b_1}x_2^{b_2}\dots x_n^{b_n}$ 

$x_i^j$ 为有 $j$ 个长度为 $i$ 的置换环的方案数，一个环上的所有结点颜色都相同。遇到具体问题时需要具体计数求解

##### 正 n 边形的旋转群的轮换指标

$P_G = \frac{1}{n}\sum_{d|n}\varphi(d)x_d^{n/d}$

##### 正 n 边形的二面体群的轮换指标

$P_G = \frac{1}{2n}\sum_{d|n}\varphi(d)x_d^{n/d} + \begin{cases} \frac{1}{2} x_1x_2^{\frac{n-1}{2}} \quad n为奇数\\ \frac{1}{4}(x_2^{\frac{n}{2}} + x_1^2x_2^{\frac{n - 2}{2}}) \quad n为偶数\end{cases}$ 

##### 顶点置换群

$P_G = \frac{1}{24}(x_1^8 + 8x_1^2x_3^2 + 9x_2^4 + 6x_4^2)$

##### 边置换群

$P_G = \frac{1}{24} (x_1^{12}+8x_3^4+6x_1^2x_2^5+3x_2^6+6x_4^3)$

##### 面置换群

$P_G = \frac{1}{24}(x_1^6+8x_3^2+6x_2^3+3x_1^2x_2^2+6x_1^2x_4)$

#### 翻转

若 $n$ 为奇数，则可以任选一个点及其对面边的中点进行翻转，除去选择的点后，我们还有 $\lfloor \frac{n}{2} \rfloor$ 个点可供自由选择（因为对称轴两边的点应该相同）， 即 $n \times cnt$ 

若 $n$ 为偶数，则可以选择两个点翻转，还剩 $\frac{n}{2} - 1$ 个点可供自由选择；还可以选择两个边的中点翻转，还剩 $\frac{n}{2}$ 个点可供自由选择，即 $\frac{n}{2} \times cnt_1 + \frac{n}{2} \times cnt_2$ 

注意枚举被选择为对称轴的点的影响



### Polya 定理

集合 $X$ 为给集合 $A={a_1, a_2, \dots a_n}$ 的每个元素赋予式样（颜色、种类等）的映射的集合

引入表示式样的集合 $B$ ，令 $X = \{f | f:A \rightarrow B\}$ ，记为 $B^A$ 

简单来说，$B$ 为题目中颜色、种类等等，$f$ 为一次操作（置换、翻转等等）

式样清单：$G$ 轨道的集合

种类的权值：假设 $B$ 上的每个元素 $b$ 都被赋予了权值 $w(b)$ 
$f\in B^A$ 的权值：$w(f) := \Pi_{a\in A}w(f(a))$ 
$G$ 轨道的权值：$w(F):=w(f)$ ，任选一个 $f\in F$ 
$B^A$ 关于 $G$ 的式样清单记为 $\mathcal{F}$ ，则：
$\sum_{F \in \mathcal{F}} w(F)= P_G(\sum_{b \in B}w(b), \sum_{b \in B}w(b)^2, \dots, \sum_{b \in B}w(b)^n)$ 

题目中我们先经过操作变换求出 $P_G(x_1, x_2, \dots, x_n)$ ，$x_i := \sum_{b \in B} w(b)^i$ ，将 $x_i$ 替换为不定元，例如 $x_1 := r, x_2:=b, \dots$ 即可将该式子转换为生成函数，答案即为某一项的值

****

## 亚线性筛

### 杜教筛

核心 $idea：$ 构造两个容易求前缀和的积性函数 $g, h$ ，使得 $h = f * g$ 

设 $S(n) = \sum_{i = 1}^nf(i)$ ，则 $S(n) = \sum_{i=1}^nh(i) - \sum_{d = 2}^ng(d)S(\lfloor \frac{n}{d} \rfloor)$ 

复杂度：$O(n^{\frac{3}{4}}) \quad n \le 10^{11}$ 

条件：需要能凑出容易求前缀和的另外两个积性函数

### Min25 筛

核心 $idea：$ 两步，求质数部分的和，求整体的和

求质数部分：

假设 $f(p) = p^k \ (p 是质数)$ ，令 $P = {p_1,p_2,\dots,p_{|P|}}$ 为 $\leq \sqrt{n}$ 的全体质数的集合，其中 $p_i$ 为第 $i$ 小的质数，特殊地，令 $p_0 = 1$ 。$min_p(x)$ 表示 $x$ 的最小质因子 。

设 $g(n, j) := \sum_{i = 2}^n[i \in P\ 或者\ min_p(i) > p_j]i^k$ ，则 $g(n, 0) = \sum_{i=2}^ni^k$ 且 $g(n,j) = g(n, j - 1) - p_j^k(g(\frac{n}{p_j},j-1) - g(p_j - 1, j - 1))$ ，$g(n,|P|)$ 就是质数部分的和

 求整体的和：

设 $h(n, j):=\sum_{i=2}^n[min_p(i) \ge p_j]f(i)$ ，则 $h(n, j)=g(n,|P|) - \sum_{k = 1}^{j - 1}f(p^k) + \sum_{k\ge j, e\geq1}(f(p_k^e)h(\frac{n}{p_k^e}, k+1) + f(p_k^{e+1}))$ ，$h(n, 1)+1$ 就是整体的和

复杂度：第一步为 $O(n^{\frac{\frac{3}{4}}{log(n)}})$ ，第二步为 $O(n^{1-\epsilon})$ 

条件：需要满足对于质数 $p$ ，$f(p)$ 是一个关于 $p$ 的多项式

****

## 二项式反演

记 $f_n$ 表示恰好使用 $n$ 个不同元素形成特定结构的方案数，$g_n$ 表示从 $n$ 个不同元素中选出 $i \geq 0$ 个元素形成特定结构的总方案数，即 $g_n$ 表示至多有 $n$ 个元素满足特定结构的方案数

$g(n) = \sum_{i=0}^{n}\big( ^n _i \big) f(i)$ 
$f(n) = \sum_{i=0}^n(-1)^{n-i} \big( ^n_i\big)g(i)$ 

考虑序列 $\{ f_n \}, \{g_n\}$ 的指数生成函数 $F(x), G(x)$ ，则 $F(x) = e^xG(x), \ G(x) = F(x)e^{-x}$ 

****

## Min-Max 反演

### 在期望意义下成立

设 $min(S) = min_{a_i \in S}\  a_i$ ，相应的 $max(S) = max_{a_i \in S} \ a_i$ 

$max(S) = \sum_{T \subseteq S} \ (-1)^{|T| + 1} \ min(T)$

$min(S) = \sum_{T \subseteq S} \ (-1)^{|T| + 1} \ max(T)$

$Kthmax(S) = \sum_{T \subseteq S} \ (-1)^{|T| - k}\ (^{|T| -1} _{k - 1})min(T)$

我们还可以得到： $lcm(S) = \Pi_{T \subseteq S} \ (gcd(T)) ^{(-1)^{|T| + 1}}$



常见题型：$n$ 个数，每单位时间每个数有概率 $p_i$ 出现，求每个数至少出现 $cnt_i$ 次的期望时间

题目所求即 $max(S)$ ，集合 $S$ 所有元素中最晚到达 $cnt_i$ 次的期望时间；$min(T)$ 则为集合 $T$ 所有元素中最早达到 $cnt_i$ 次的期望时间，即求 $T$ 的所有状态的期望值之和，满足对于 $T$ 的每个元素 $0, 1, \dots, m$ ，其出现次数为 $c_0 < cnt_0, c_1 < cnt_1, \dots,c_m < cnt_m$ 

****

## Prufer 序列

初始时为空序列，如果当前树上多于两个结点，假设当前标号最小的叶子为 $x$ ，与 $x$ 相连的结点标号为 $y$ ，则把结点 $x$ 从树上删掉，把 $y$ 放到序列的末尾，不断重复，最后生成的序列即为 $Prufer$ 序列

$n$ 个点的有标号无根树共有 $n^{n - 2}$ 颗

度数为 $d$ 的节点会在 $Prufer$ 序列中出现 $d - 1$ 次

****

## 矩阵树定理

设无向图 $G(V, E)$ ，$D = diag(d(1),d(2),\dots,d(n))$ ，$d(i)$ 是结点 $i$ 的度数（若存在边权，则为所有连向该点的边权的和，若为有向图，则度数为入度），$G$ 的拉普拉斯矩阵 $L = D - E$ 。$G$ 的生成树个数（权值）为 $det(L_0)$ ，其中 $L_0$ 是 $L$ 去掉第 $i$ 行第 $i$ 列（$i$ 任选）

****

## LGV 引理

定义：

$\omega(P)$ 表示 $P$ 这条路径上所有边的边权之积。（路径计数时，可以将边权都设为 $1$ ）（事实上，边权可以为生成函数）

$e(u,v)$ 表示 $u$ 到 $v$ 的每一条路径 $P$ 的 $\omega(P)$ 之和，即 $e(u,v) = \sum_{P:u\rightarrow v} \omega(P)$ 。

起点集合 $A$ ，是有向无环图点集的一个子集，大小为 $n$ 。

终点集合 $B$ ，也是有向无环图点集的一个子集，大小也为 $n$ 。

一组 $A \rightarrow B$ 的不相交路径 $S$ ：$S_i$ 是一条从 $A_i$ 到 $B_{\sigma(S)_i}$ 的不相交路径（$\sigma(S)$ 是一个排列），对于任何 $i \neq j$ ，$S_i$ 和 $S_j$ 没有公共顶点。

$t(\sigma)$ 表示排列 $\sigma$ 的逆序对个数



引理：

$M = \begin{bmatrix} e(A_1,B_1) & e(A_1,B_2) & \cdots & e(A_1,B_n) \\ e(A_2,B_1) & e(A_2,B_2) & \cdots & e(A_2,B_n) \\ \vdots & \vdots & \ddots & \vdots \\ e(A_n,B_1) & e(A_n,B_2) & \cdots & e(A_n,B_n) \end{bmatrix}$ 

$det(M) = \sum_{S:A\rightarrow B}(-1) ^ {t(\sigma(S))} \Pi_{i=1}^n \omega(S_i)$ 

其中 $\sum_{S:A\rightarrow B}$ 表示满足上文要求的 $A \rightarrow B$ 的每一组不相交路径 $S$ 

即左边行列式的值等于固定起点集合，排列终点集合所形成的所有不相交路径的带权和

在边权为 $1$ 的图中， $e(u, v)$ 为 $u \rightarrow v$ 的路径数目



常用情况：排列形成的不相交的路径对是唯一的，即只有唯一的 $\sigma$（对其他的映射都不成立）

****

## 马尔科夫链

$Q$ 为转移矩阵，$p_{jk}$ 为 $j \rightarrow k$ 的转移概率

### 稳态收敛定理：

对于每个 $j$ ，我们有： $\forall i \ \lim_{n \rightarrow \infty} r_{ij}(n) = \pi_j$ 

$\pi_j = \sum_{k= 1}^{m}\pi_kp_{kj} = 1,\ j = 1,\dots,m$ 

$\sum_{k=1}^{m}\pi_k = 1$

### Expected Number of  Visits to Transient States

$F = (I - Q)^{-1}$ ，$F_{ij}$ 即从某个初始瞬时态 $i$ 到最终瞬时态 $j$ 的期望访问次数

考虑一个马尔科夫链的 $n$ 次转移，该链是从给定初始状态出发的、非周期的，且具有单个常返类。令 $q_{jk}(n)$ 为在时间 $n$ 内，从状态 $j$ 到状态 $k$ 的转移期望次数，那么无论初始状态是什么，均有：$\lim _{n \rightarrow \infty} \frac{q_{jk}(n)}{n} = \pi_j p_{jk}$ 

****

## 鞅与停时定理

鞅是一个特殊的随机过程，其未来的值的期望等于当前的值

停时定理：在一个随机时刻停止观察鞅时，鞅的期望保持不变

我们需要构造一个势能函数，势能的变化量恒定为 $1$ ，每操作一次后，势能函数都会变化，直到减少到某一个常数停止，而减少的次数就是期望时间

即：如果我们能给状态 $S$ 设定一种函数 $F(S)$ 使得每一次操作都会使 $F(S)$ 的期望增加 $1$ ，且终止态无法继续转移，那么我就可以用 $F(S_{end}) - F(S_{begin})$ 来表示从当前态到终止态的期望（注意考虑变量之间的独立性，即若函数中有多个变量，考虑是否可以分离单独计算）

构造下列式子：$E(\Phi(A_{n + 1})= - \Phi(A_{n})| A_0, A_1, \dots, A_n) = -1$ 
即构造两个函数，$f(state_1) - f(state_0) = -1$ 
