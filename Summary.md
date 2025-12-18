## 典题

https://codeforces.com/gym/103098/problem/L
$3 \times n$ 的网格
转移时考虑对每个状态进行翻转操作，状态依旧等价，再进行转移，即可不重不漏

https://www.luogu.com.cn/problem/AT_arc101_d
给定数组 $r$ ，每个位置可以填 $0$ 或 $1$ ，满足 $i < j, r[i] > r[j]$ ，若 $i$ 位置为 $1$ 则 $j$ 位置一定为 $1$ 
顺序枚举，考虑 $dp$ 最后一个可自由选择的 $1$ 的位置，则 $dp[i] = 1 + \sum_{j < i \ and \  r[j] < r[i]} dp[j]$ 

https://ac.nowcoder.com/acm/contest/32708/E

给定一个字符串，多组询问，每次询问删除至少包含 $[l, r]$ 这段区间的字符串后，可以形成多少个本质不同的字符串，形式化的讲：对于 $nl \leq l, nr \geq r$ ，求删除 $[nl, nr]$ 后可以形成多少个本质不同的字符串
如果删除 $[l, r]$ 这段区间后形成的字符串，与删除 $[l + k, r + k]$ 这段区间后形成的字符串都相同，则对于 $i \leq k$ ，删除 $[l+i, r+i]$ 与删除 $[l, r]$ 后形成的串也都相同，考虑 $[l, r]$ 与 $[l + 1, r + 1]$ ，即当 $s[l] == s[r + 1]$ 时满足，即每当有一个 $s[l] == s[r]$ 时，所形成的子串数量都会减少 $1$ ，所以我们便可以计数，使用莫队即可

https://codeforces.com/contest/2127/problem/F

计数问题中，考虑每种不同序列中的每个位置的和是否相同（即对任意的 $i, j$ 交换是否构成一个新序列），如果相同，那么每个位置的贡献都为等价的，即可简化计数内容

https://codeforces.com/gym/105615

给定 $x$ 和 $y$ ，每次可以 $x \leftarrow x + 1$ 或者 $x \leftarrow x \times 2$ ，求方案数
考虑枚举 $\times 2$ 操作的次数 $t$ 。对于 $+1$ 操作，如果之前有 $i$ 次 $\times 2$ 操作，那就等价于 $+2^i$ 。所以不同操作序列的数量相当于将 $y - 2^tx$ 拆分为若干个 $2^0, \dots, 2^t$ 之和的不同拆分方案的数量。
考虑如何计数，钦定拆分方案按照次幂降序排列，设 $f(i, j, k)$ 表示 $2^i$ 拆分为若干 $2$ 的次幂，最高次不超过 $j$ ，且最低次幂恰好为 $k$ 的方案数。显然有：$f(i,j,k) = \sum_{x=k}^{j}f(i-1,j,x)f(i-1,x,k)$
将 $y-2^tx$ 写作二进制，从高位到低位考虑是 $1$ 的二进制位，利用 $f$ 的信息转移。具体来说，设 $g(i, j)$ 表示当前从高位到低位考虑到 $2^i$ ，当前拆分方案最低次幂为 $j$ 的方案数。初始时 $g(N, t)=1$ ，因为 $2^tx$ 将 $x$ 看作系数 $1$ ，即把 $2^t$ 放入序列中的第一个，且是最大值，因为最多 $\times 2$ 了 $t$ 次。若 $y-2^tx$ 二进制下第 $i$ 位为 $0$ ，则 $g(i, *) = g(i+1, *)$ 。否则：$g(i, j) = \sum_{k \geq j}g(i + 1, k)f(i, k, j)$ 
最终 $\sum g(0, * )$ 即为答案

https://www.luogu.com.cn/problem/P2824

每次操作区间排序，求最后的序列。维护每个有序序列，并标记升序降序，在线线段树分裂合并求第 $k$ 大即可。

https://codeforces.com/contest/2085/problem/F2

每次操作可以交换数组中相邻两个数，考虑使某个满足条件的子数组 $(len = n)$ 操作步数最小，一定是枚举一个不动点，左边选择 $\frac{n}{2}$ 个点移动，右边再选择 $\frac{n}{2}$ 个点移动，往不动点靠拢，计算到不动点的距离即可

https://codeforces.com/contest/1830/problem/F

$dp$ 方程设置为第 $i$ 个点为结尾，但 $i$ 点的贡献还没有计算的最大值，$dp_i = max(dp_j + 区间个数(j, i) \times p_j)$ ，于是可以用 $KTT$ 动态维护，从左到右枚举，每个区间结束时将还没计算的贡献区间加上

https://codeforces.com/contest/2056/problem/F2

组合计数异或求值，考虑 $\% 2$ 意义下的组合数和斯特林数，考虑对贡献时将数按相邻两个分组（每组内异或为 $1$ 

https://acm.hdu.edu.cn/showproblem.php?pid=4747

求所有区间的 $mex$ 
因为区间内 $mex$ 每次增加一个数，不好直接计算，考虑对每个区间进行删数，维护每个以 $i$ 为右端点的区间 $mex$ ，从左往右每次删除一个数，由于区间长度越来越小，且为包含关系，即 $mex$ 满足单调性，线段树上二分区间设置为一个数即可

https://codeforces.com/contest/2075/problem/F

首先观察出满足答案的左右端点的移动具有单调性，二维偏序关系用 $heap$ ，每次将小于当前值的数和位置都扔到 $heap$ 中，$heap$ 以位置作为关键字排序，再将位置不满足的进行弹出，线段树上维护以某个位置为右端点的最长值，枚举作为左端点的位置统计答案即可

https://codeforces.com/contest/1823/problem/F

树上随机图游走，朴素方程为循环依赖的 $dp$ 
树上 $trick$ ：任意一个点 $u$ ，其父亲节点为 $p$ ，其 $dp_u$ 都可以表示为 $dp_p * a_u + b_u$ ，其中 $a_u, b_u$ 为待定系数，随后即可进行树形 $dp$ ，注意边界情况

[Problem - G - Codeforces](https://codeforces.com/contest/1418/problem/G)

求满足区间内每个数出现次数都为 $k$ 的区间个数，将问题拆分为两部分：

1. 区间每个数出现次数为 $k$ 的倍数  $\rightarrow$  前缀和哈希，每个数前缀的出现次数对 $k$ 取模
2. 区间每个数出现次数不超过 $k$   $\rightarrow$  双指针，若当前值的出现次数超过 $k$ ，则移动另一个指针

证明 $a = b$ 可以证明 $a \leq b \ \& \ a\ge b$ ，对于最后一个数的前缀 $+1$ ，倒数第 $k$ 个数的前缀 $-1$ ，判断 $0$ 的个数

每次叉掉出现次数 $<k$ 和 $> k$ 的区间，随后求矩阵面积并

https://codeforces.com/contest/1523/problem/H

每个 $i$ 可以跳到 $[i, a_i]$ 位置，多组询问，求删除 $k$ 个点，最少跳多少次可以从 $l$ 跳到 $r$ 

显然观察，每跳一次一定会跳到 $j \in [i, a_i]$ 中 $j + a_j$ 最大的，每往后面删除一个数，范围会再扩大 $1$，考虑倍增 $dp$ ，在 $i$ 位置一共删了 $j$ 个数，跳了 $2^k$ 最远能到哪个点，每次询问时倍增答案即可。

[Problem - F - Codeforces](https://codeforces.com/contest/1295/problem/F)

考虑对区间离散化，枚举 $i$ 前有几个点一起在第 $j$ 个区间，区间外的点直接统计答案，区间内的利用组合计数统计

[G - XOR Neighbors (atcoder.jp)](https://atcoder.jp/contests/abc366/tasks/abc366_g)

当集合具有异或性质时，将集合作为一个二进制数插入到线性基中

https://codeforces.com/gym/105176 $J$ 题：随机化，将原数组打乱进行 $dp$ 

在某些背包问题中，我们需要解决一些零和问题，即：物品有正有负且都不是特别大（$|w_i| \leq W$ ），我们需要找到一个具有某种性质的子集 $S$ ，满足 $\sum_{S}w_i=0$ 。

对容量这一维进行 $dp$ ，体积有可能达到 $nW$ 级别，但如果问题只需要和答案有关的某一个 $instance$ 被 $dp$ 过程命中的话（更具体地说就是比如问存在性或者某种最优性，那我们只需要达到最优解的那个方案被统计到），我们有这样的一个想法：在 $dp$ 之前就把物品随机打乱，那我们所需的那个 $instance$ 的前缀和总是不会很大。这样以来，我们就可以将这一维背包的容量缩小

[Problem - E - Codeforces](https://codeforces.com/contest/643/problem/E)

当期望题目要求输出小数而非取模时，则需要考虑某些情况的概率是否可以被忽视掉

题目中即使子树很深，随机删边后深度依然很大的概率非常小，具体地，其深度大于 $60$ 的概率远小于所提供的误差 $10^{-6}$ ，所以我们只需要考虑深度 $\leq 60$ 的情况即可

[Problem - G - Codeforces](https://codeforces.com/contest/643/problem/G)

线段树上拓展摩尔投票

[Problem - D - Codeforces](https://codeforces.com/contest/2005/problem/D)

前后缀 $gcd$ ，合并以 $l$ 为左端点或者以 $r$ 为右端点 $gcd$ 的 $log(V)$ 个段

不要忘记 $gcd$ 函数时间复杂度为 $log$ ，易忘记考虑导致超时

https://codeforces.com/contest/121/problem/D

双指针加快速统计答案，对于 $n$ 个区间的答案，考虑加入区间限制和左右端点，利用对答案的贡献来进行 $O(1)$ 计算

[P3813 [FJOI2017] 矩阵填数 - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/P3813)
子矩阵容斥原理

https://codeforces.com/problemset/problem/1758/E

给定一个矩阵，有一些格子的数没有确定，每次可以给一行或一列 $+1$ 或 $-1$ （在模 $h$ 的条件下），求有多少个方案数使得矩阵全为 $0$ 

考虑图论，若格子的数是确定的，则添加 $i$ 行到 $j$ 列的边，边权为数的值，每次假定某行或某列的初始贡献为 $0$ ，去计算其他行或列的贡献，若合法则可以将其合并，若最后联通量个数为 $cnt$ ，则答案为 $h^{cnt-1}$ 

https://codeforces.com/contest/1774/problem/G

给定 $n$ 个区间和 $q$ 个询问，每次询问只用奇数个区间拼成 $[l, r]$ 的计数方案减去偶数个区间拼成 $[l, r]$ 的计数方案

考虑对答案的贡献，若区间 $X$ 包含区间 $Y$ ，则选择 $X$ 后，$Y$ 可选也可不选，对答案无影响，则我们可以强制 $X$ 不选

$hint$ ：答案只可能为 $-1, 0, 1$ 

故我们去掉所有相互包含的区间再排序，即有若$l_i < l_j$ 则 $r_i < r_j$，然后找出左端点大于等于 $l$ 的前两个区间，同时进行倍增（对于每个 $i$ ，跳到下一个 $l_j > r_i$ 的位置）

观察最后能构成 $[l, r]$ 的合法方案，一定是最初的两个区间不会跳到同一个区间上，则最后即可判断答案

https://www.luogu.com.cn/problem/CF1025G

势能函数求期望，鞅，停时定理

考虑设计一个函数将一个局面到最终局面的期望进行评估，如果这个函数能够满足任意一次操作会使得函数值的期望增加 $1$，那么就可以用最终局面的函数值减去初始局面的函数值就得到期望操作次数。同时，必须满足最终局面无法转移。



## CF927

[Problem - G - Codeforces](https://codeforces.com/contest/1932/problem/G)

考虑时间对图的影响，对于 $dijkstra$ 而言，若同一个点 $u$ 的某个时间 $t_i < t_j$ ，不存在 $t_j$ 能比 $t_i$ 使得另一个 点 $v$ 更新的时间更优，则可以直接使用

https://codeforces.com/contest/295/problem/D

当朴素 $dp$ 状态与两个端点相关时，考虑是否可以将端点的两维优化为一维的长度

## $ABC342$

[E - Last Train (atcoder.jp)](https://atcoder.jp/contests/abc342/tasks/abc342_e)

正权边求最长路也可直接使用 $dijkstra$ 

[F - Black Jack (atcoder.jp)](https://atcoder.jp/contests/abc342/tasks/abc342_f)

首次遇到两个参数的概率 $DP$ ，需要预处理其中一个参数的分布概率

[G - Retroactive Range Chmax (atcoder.jp)](https://atcoder.jp/contests/abc342/tasks/abc342_g)

树套树配合标记永久化



## $CF 929$

[Problem - F - Codeforces](https://codeforces.com/contest/1933/problem/F)

对于图上障碍物也在移动的问题，可以将障碍物视为不动，来计算当前单位对于障碍物的相对移动，同时要注意终点也会移动

[Problem - G - Codeforces](https://codeforces.com/contest/1933/problem/G)

当限制非常强的情况下，可以先打表求出题目要求的最小 $n, m$ 时所有的合法情况，下面则考虑一般情况对于所搜索出来的合法情况的扩展，大胆猜测一般情况，随后则可进行找规律 $/$ 推通项



## $CF930$

[Problem - D - Codeforces](https://codeforces.com/contest/1937/problem/D)

求多个往返点的往返距离，可以考虑用队列维护往返点，每次向一个方向扩展一次时，增加的往返距离的贡献为往返点的个数 $\times 2$ 

[Problem - E - Codeforces](https://codeforces.com/contest/1937/problem/E)

前后缀优化建图（指一个点连向的区间一定是一段前缀或一段后缀的形式，将连向一段区间的边优化为了连向一个或两个点或几个点的边），将每个点的每个属性也设置为图中的一个点，随后按每个属性值大小排序，相邻两个点依据差值连边，优化朴素连边的数量

$PS:$ 之前某个 $ABC$ 的 $G$ 最大权闭合子图遇到过此种连边方式（

[Problem - F - Codeforces](https://codeforces.com/contest/1937/problem/F)

拆位加最大子段和的思想，考虑横跨两个区间的贡献，从高位到低位枚举，考虑每一位取不取，若当前该位 $v$ 值存在，则必须取；否则若取，则比 $v$ 更大，不取则与 $v$ 相同，继续考虑下一位。对于左右区间应该选择取后使答案更小的那一侧端点



## $ABC343$

[G - Compress Strings (atcoder.jp)](https://atcoder.jp/contests/abc343/tasks/abc343_g)

在 $N$ 的个数较小的情况下，多个串判断子串也可用 $KMP$ ，不要只限制在 $ACAM$ 上



## $CF236$

[Problem - 402E - Codeforces](https://codeforces.com/problemset/problem/402/E)

邻接矩阵 $A$ ，$A^k$ 中若 $A^k[i][j]$ 为 $1$ ，则说明 $i$ 与 $j$ 之间存在一条长度为 $k$ 的路径



## $CF932$

[Problem - C - Codeforces](https://codeforces.com/contest/1935/problem/C)

树状数组求最大前缀和小于 $K$ 

同值非相同离散化，严格离散到 $[0, n)$ 

[Problem - E - Codeforces](https://codeforces.com/contest/1935/problem/E)

对于有上下界的位运算相关题目，首先考虑求最大相同前缀（因为其高位的贡献总是比低位更大，优先考虑高位），将上下界减去 $lcp$ ，再按位考虑（选择当前位或补充低位等常见位运算操作）

[Problem - F - Codeforces](https://codeforces.com/contest/1935/problem/F)

对于集合中包括多个离散的值时，优先考虑仅用最大值 / 最小值是否能构造或解题，将离散的数值转化为连续的区间，参考线段区间合并时考虑左右端点，等价于此处的最大最小值



## $ABC 344$

[F - Earn to Advance (atcoder.jp)](https://atcoder.jp/contests/abc344/tasks/abc344_f)

$dp$ 的值存在二维，贪心地去证明二维值的更新条件



## $CF933$

[Problem - G - Codeforces](https://codeforces.com/contest/1941/problem/G)

每个点的状态数等于与该点链接的边的颜色数，故所有点的状态数之和为 $O(m)$ ，直接用 $map$ 存状态即可

设立一个虚拟颜色，当需要向不同颜色的边转移时，先向虚拟颜色点转移，再有虚拟颜色点转移到其他颜色； 当需要相同颜色的边转移时，直接枚举当前点该颜色的所有边。即虚拟颜色点只枚举颜色状态，非虚拟颜色点只枚举边状态，转移复杂度为 $O(颜色数) + O(边数)$ 



## $ABC345$

[E - Colorful Subsequence (atcoder.jp)](https://atcoder.jp/contests/abc345/tasks/abc345_e)

最优解一定是当前价值最大的两种颜色之一转移过来的，当时间复杂度不足时，多去猜想性质和结论



## $ARC174$

[C - Catastrophic Roulette (atcoder.jp)](https://atcoder.jp/contests/arc174/tasks/arc174_c)

多人交替进行操作的概率期望 $dp$ ，设计状态多加一维，即当轮到第 $i$ 个人时的期望



## $CF936$

[Problem - D - Codeforces](https://codeforces.com/contest/1946/problem/D)

$w_1 | (w_1 \oplus w_2) | (w_2 \oplus w_3) |  \dots | (w_{n-1} \oplus w_n) = w_1 | w_2 | w_3 | \dots | w_n$  

[Problem - F - Codeforces](https://codeforces.com/contest/1946/problem/F)

对于多询问，首先考虑单次询问 $[0, n)$ 的情况，求出平凡的递推数组，对于单个 $[l, r)$ ，考虑固定某一个端点，例如本题中的 $dp$ 状态为，$dp[a_p]$ 指以 $a_p$ 为结尾，起点为 $l$ 的满足条件的个数，倒序枚举 $l$ ，因为 $a_p$ 为终点，所以在每个 $l$ 循环中，$dp[a_p]$ 维护的值便是以 $a_l$ 为起点，以 $a_p$ 为结尾的所有个数，$\sum_{i=l}^p dp[a_p]$ 则为 $a_p$ 在当前 $l$ 循环中的所有贡献，$\sum_{i=l}^{r - 1}\sum_{j=l}^idp[a_i]$ 即为 $[l, r)$ 的查询答案，树状数组单点维护 $\sum_{i=l}^p dp[a_p]$ ，区间查询即可



## $CodeTON \ Round \ 8$

[Problem - D - Codeforces](https://codeforces.com/contest/1942/problem/D)

考虑 $dp$ 是否具有凸性，即如果第 $k$ 大没被选，那么第 $k + 1$ 大也不会选

对于每一个独立的段，考虑先将最大的待选 $dp$ 值与该段加入到堆中，当其被弹出时再将次大值与该段加入到堆，以此重复 $k$ 次



## $CF 914$

[Problem - E - Codeforces](https://codeforces.com/contest/1904/problem/E)

树的直径性质：

当合并两个两棵树时，新的树的直径的两个端点，一定是在原来两棵树中直径的四个点里选两个点

$x$ 点所在连通块能到的最远点，一定是 $x$ 这个连通块的直径的两个端点中的一个

随后便可转化成 $dfs$ 序上的区间合并问题



## $CF \ Global \ Round \ 25$

[Problem - 1951E - Codeforces](https://codeforces.com/problemset/problem/1951/E)

构造一种方案，将一个字符串划分为多个非回文串。结论：若存在一个合法方案，则一定可以划分为 $2$ 段非回文串

[Problem - F - Codeforces](https://codeforces.com/contest/1951/problem/F)

考虑已知的排列 $p$ 对于 $inv(q) + inv(q * p)$ 的影响，即枚举 $p$ 中的逆序对情况，随后检查 $q$ 和 $q * p$ 的逆序对情况

之后将式子代换，尽量将式子变换出已知的 $p$ 



## $ABC348$

[G - Max (Sum - Max) (atcoder.jp)](https://atcoder.jp/contests/abc348/tasks/abc348_g)

决策单调性！！！

优先考虑单个 $k$ 时如何做，若无法直接优化，则考虑答案的选择是否有决策单调性，通过简单证明和打表进行验证，随后用分治即可，参数中传入当前求的 $m$ 可考虑的选择区间



## $CF864$

[Problem - 1797F - Codeforces](https://codeforces.com/problemset/problem/1797/F)

点权 $Krustal$ 重构树，每次直接将未连通的两个点 $merge$ ，将最大 / 最小的点作为父节点，将另一个节点直接挂下面即可



## $EDU \ CF \ 165$

[Problem - E - Codeforces](https://codeforces.com/contest/1969/problem/E)

求存在一个数只出现一次的区间个数，考虑每个数的出现位置，其贡献范围为上一次出现的位置到下一次出现的位置，当固定区间左端点后，用线段树动态维护区间加，即可计算出每个合法区间。$dp[i]$ 为以 $i$ 为左端点，第一个不满足的右端点在哪里

相同 $trick$ 题目：[G - Alone (atcoder.jp)](https://atcoder.jp/contests/abc346/tasks/abc346_g)



## 牛客小白月赛 $92$

[G-不速之客_牛客小白月赛92 (nowcoder.com)](https://ac.nowcoder.com/acm/contest/81126/G)

另类数位 $meet \ in \ middle$ ，对于复杂函数即可考虑此类优化搜索做法



## $ABC352$

[G - Socks 3 (atcoder.jp)](https://atcoder.jp/contests/abc352/tasks/abc352_g)

一旦有相同颜色即停止 $\Rightarrow$ 每种颜色出现一次或零次

$f[i]$ 即手中有 $i$ 种颜色袜子的所有出现次数，除以当前所有的可能情况，即抽取第 $i$ 轮的合法概率



[F - Estimate Order (atcoder.jp)](https://atcoder.jp/contests/abc352/tasks/abc352_f)

考虑每个连通块内部的排名情况，随后对当前连通块之外的部分进行 $dp$ ，对排名进行状压



## $2023$ 济南区域赛

https://codeforces.com/gym/104901/problem/E

给定一张平衡二分图。您需要添加恰好一条边，连接 $U$ 中的一个节点与 $V$ 中的一个节点，使得图的匹配数增加，求方案数。

建源点 $S$ 向二分图左侧每个点连流量为 $1$ 的边，汇点 $T$ 从二分图右侧每个点连流量为 $1$ 的边，原图的匹配数就是新图的最大流。

先求出新图的最大流以及对应的残量网络，添加一条边能够提高匹配数当且仅当新的残量网络上源 $S$ 和汇 $T$ 联通。因此求出源 $S$ 能到达左侧多少点，右侧多少点能达到汇 $T$ ，相乘即为答案。



## $2024$ 东北邀请赛

[Dashboard - The 2024 CCPC National Invitational Contest (Northeast), The 18th Northeast Collegiate Programming Contest - Codeforces](https://codeforces.com/gym/105173)

$I$ : $dp$ 组合计数问题

$f[i]$ 为最后一个完整排列的结尾在 $i$ 的方案数

$g[j]$ 为在一个完整排列后接上 $j$ 个数，满足除了接上的第 $j$ 个位置满足为一个排列，其他接上的位置都无法与前面的数形成排列。

考虑用排列 $1,2,\dots,k$ 来递推 $g[j]$ ，考虑容斥减去不合法的位置，对于一个不合法的排列，它可能存在若干个前缀符合 $[1, i]$ 是一个 $1$ 到 $i$ 的排列（因为这样会与前面原有排列的 $i+1$ 到 $k$ 形成一个新排列），那么我们枚举每一个不合法排列最后一个违反限制的前缀，在这个位置将其减去。

假设当前枚举到 $i$ ，对于 $i + 1$ 到 $j$ 这部分，由于我们钦定 $i$ 是最后一个违反限制的前缀，故 $[1, i]$ 与 $[i + 1, j]$ 不能再违反限制，即对于任意 $i < p < j$ ，$[i + 1, p]$ 这一段不能是 $i + 1$ 到 $p$ 的一个排列，即转化为了子问题。

$g_i = i!-\sum_{j=1}^{i-1}j! \times g_{i-j}$ 

$f_i = \sum_{j=1}^{k}f_{i - j} \times g_j$ 

重点思想：考虑用最平凡的排列 $1, 2, \dots, k$ 去思考任意一个排列；考虑状态设置为最后一个完整排列，因为最后答案的合法方案，一定是被最后一个完整排列覆盖，倒推状态设置，再去考虑递推公式



## $2024$ 昆明邀请赛

[Problem - L - Codeforces](https://codeforces.com/gym/526652/problem/L)

把所有斜线按 $x_i$ 升序为第一关键字，$y_i$ 降序为第二关键字排序，走到 $(x, y)$ 最多经过几条斜线，就是满足 $x_i + 1 ≤ x$ 且 $y_i + 1 ≤ y$ 的最长上升子序列的长度。

算最长上升子序列的时候，我们要维护一个二分数组 $f_i$，表 示当前 $LIS$ 长度是 $i$ 的最小元素是多少。对应到本题里，就是当前最多经过 $i$ 条斜线的 $y$ 最小是多少。所以对于固定的 $x$，答案就是 $∑(q − f_i)$。随着 $f$ 值的变化维护这个和即可。

## $2024$ 哈尔滨区域赛

期望 $E(x^y) = \sum{p(i_1,i_2,\dots,i_y)}$ ，即 $y$ 个各不相同的点同时出现的概率

对于线性序列上的问题而言，以 $E(x^3)$ 举例，考虑增量使得 $x+1$ ，即可递推出 $E((x + 1) ^ 3)$ 与 $E(x^3)$ 之间的关系，再进行 $dp$ 

## $2024$ 南京区域赛

https://qoj.ac/contest/1829/problem/9565

给一串包含 $0,1,2$ 的字符串，相邻的 $0$ 之间或者 $1$ 之间可以抵消，$2$ 可以变换为任意的 $0$ 或 $1$ ，求字符串最短为多少（Trick！）

考虑 $erase$ 操作时，影响的一定是原串中奇偶不同的位置下标，考虑让消除操作只跟奇偶位置有关，将所有偶数位置的 $0$ 或 $1$ 取反来转化，即奇偶位置的数不同可进行消除

https://codeforces.com/gym/105484/problem/C

拓扑序计数：给定一棵树，在点上写一个排列，使得每个点的点权比其父亲的数大的方案数，等价于满足每个节点的父亲排在他前面的排列方案数，为：$\frac{n!}{\Pi_{i=0}^{n-1}{siz[i]}}$ 

## $CF495$

[Problem - 1004F - Codeforces](https://codeforces.com/problemset/problem/1004/F)

线段树上计数，维护 $O(logV)$ 个前后缀 $or$ 和



## $CF949$

[Problem - E - Codeforces](https://codeforces.com/contest/1981/problem/E)

考虑所有可能的边中哪些是侯选边，当连续三个递增的 $a_i$ 相交时，判断哪些边会保留，即在递增关系上，考虑相邻位置。

随后扫描线选出所有可能的侯选边，构建最小生成树



## $CF955$

[Problem - D - Codeforces](https://codeforces.com/contest/1982/problem/D)

$c_{1} \cdot d_{1} + c_{2} \cdot d_{2} + \dots + c_{q} \cdot d_{q} = D$ 若有解

则 $$ D \bmod gcd(d_{1}, d_{2}, \dots, d_{q}) = 0 $$ 

[Problem - E - Codeforces](https://codeforces.com/contest/1982/problem/E)

对于朴素做法考虑分治，随后数位 $dp$ ，对于每个 $[0, 1 <<n)$ 记录满足条件前 $l$ 个数和后 $r$ 个数，随后考虑合并即可



## $EDU \ CF\ 167$

[Problem - E - Codeforces](https://codeforces.com/contest/1989/problem/E)

考虑 $a$ 的每种情况是否能唯一对应一种方案的 $b$ ，这里考虑对于 $a$ 的连续段

## $CF 977$

[Problem - C2 - Codeforces](https://codeforces.com/contest/2021/problem/C2)

当答案与元素的相对排序位置相关时，统计 $n$ 个元素的相邻位置大小关系即可，若 $n - 1$ 个大小关系全部满足，则答案正确

https://codeforces.com/contest/2021/problem/D

这类题目考虑将区间收拢到每一个出现的点上，此题中考虑将区间收拢到左右端点处

注意到若当前区间选择了 $[l, r)$ ，则下个区间，要么左端点 $nl \in [0, l)$ ，要么右端点 $nr \in [r, m)$ ，则设置状态为 $dpl[i]$ 即左端点必选 $i$ 的最大值和 $dpr[i]$ 即右端点必选 $i$ 的最大值。

## $EDU \ CF \ 134$

https://codeforces.com/problemset/problem/1721/F

最大独立集 = 总点数 - 最大匹配数，每次删去一个点且匹配数减少，即删除非最大独立集上的点。

最大独立集 = 最小点覆盖的补集，即求出最小点覆盖上的点删去即可



## $ABC376$

https://atcoder.jp/contests/abc376/tasks/abc376_g

典型树上贪心，这类问题的特征：从根节点开始，每次选一个跟已经遍历的点相连的点遍历，求一下值

总体做法：先思考集合上怎么做；然后考虑两个集合的先后关系；随后用数据结构维护连通块的关系

## $CF1008$

https://codeforces.com/contest/2077/problem/C

当直接求解答案难以计算时，考虑计算答案的增量变化。观察推式子发现，本题所求的答案只与字符串中 $0$ 和 $1$ 的数目有关，故可以无视位置信息，直接求出答案的变化量

## $CF1012$

https://codeforces.com/contest/2089/problem/C1

$l$ 个钥匙，$n$ 个人，每次按顺序等概率开锁，按最优方式开，求每个人开锁的期望次数

考虑前 $i$ 个钥匙第 $j$ 个人的期望次数，枚举第 $k$ 个人首次打开第 $i$ 个钥匙，第 $k$ 个人之后的就可以从第 $i - 1$ 层递推，$dp$ 即可



## 正睿 $2024$ 省选

### 组合数学

[P8367 [LNOI2022\] 盒 - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/P8367)

每进行一次操作，相当于经过一条边，令 $c, d$ 分别为 $a, b$ 的前缀和，那么答案为 $\sum_{i=1}^{n-1}w_i|c_i-d_i|$ 

考虑扩展 $i \rightarrow i + 1$ 对式子的影响（参考莫队）

$f(n,m,i,k) = \sum_{j=0}^k \big( ^{i+j-1}_{i-1} \big) \big(^{n+m-i-j-1} _{n-i-1}\big)$ 

$g(n,m,i,k) = \sum_{j=0}^k j\big( ^{i+j-1}_{i-1} \big) \big(^{n+m-i-j-1} _{n-i-1}\big)$ 

观察式子，用 $f$ 去表示 $g$ 

考虑组合意义，思考 $i$ 和 $k$ 增加时对 $f$ 的影响，$f(n,m,i,k)$ 的组合意义是前 $i$ 个数和 $\leq k$ 且所有 $n$ 个数和为 $m$ 的方案数

即从左往右的第 $k + 1$ 个小球不在前 $i$ 个盒子里，枚举第 $k+1$ 个小球放在哪个盒子，可以得到：$f(n,m,i,k) = \sum_{j=i+1}^n \big( ^{j+k-1}_{j-1} \big) \big(^{n+m-j-k-1} _{n-j}\big)$ 



## $2023 \ ICPC \ HangZhou$

[Problem - E - Codeforces](https://codeforces.com/gym/104976/problem/E)

对于前缀长度为 $l$ 的限制，可以按 $l$ 从小到大进行处理，即在 $l_i$ 的基础上去尝试是否能叠加 $l_{i+1}$ 的限制



## $2024$ 牛客多校第一场

[D-XOR of Suffix Sums_2024牛客暑期多校训练营1 (nowcoder.com)](https://ac.nowcoder.com/acm/contest/81596/D)

判断 $x - y$ 在第 $i$ 位上为 $1$ ，即判断 $(x - y)\  \% \ 2^{i+1} \geq 2^i$ ，易发现 $y$ 为连续的段

即如果固定 $x$ ，则去判断 $y$ 的范围。在本题中，设 $a$ 为 $x \  \% \ 2^{i+1}$ 的值，设 $b$ 为 $a$ 前 $i - 1$ 位的值，若 $x$ 在第 $i$ 位为 $1$ ，则 $y$ 的范围为 $[0, b + 1)$ 和 $[a + 1, 2^{i + 1})$ ；若 $x$ 在第 $i$ 位为 $0$ ，则 $y$ 的范围为 $[a + 1, a + 2^i+1)$ 



## $EPIC\ Institute\ of\ Technology\ Round\ August\ 2024\ (Div. 1 + Div. 2)$

[Problem - D2 - Codeforces](https://codeforces.com/contest/2002/problem/D2)

判断一个序列是否为 $dfs$ 序，只需判断对于每个 $0 < i < n$ ，$p[i - 1]$ 与 $p[i]$ 的 $lca$ 是否为 $p[i]$ 的父亲，若是，则 $i$ 号点成立，若 $n$ 个点都成立，则该序列为 $dfs$ 序列

## $2024\ ICPC\ Asia\ EC$ 网络预选赛第一场

[The 2024 ICPC Asia East Continent Online Contest (I) - Dashboard - Contest - QOJ.ac](https://qoj.ac/contest/1794)

给定 $n$ 个区间 $[l_i, r_i]$ ，求有多少个 $1$ 到 $n$ 的排列 $p$ 满足 $p_i \in [l_i, r_i]$ ，求答案的奇偶性

根据限制条件构造矩阵 $A$ ，使得：$A_{i, j} = [j \in [l_i, r_i]]$ 。要求的答案即 $\sum_p \Pi_{i=1}^{n} A_{ip_i}$ 

对于行列式而言：$det(A) = \sum_p sgn(p) \Pi_{i=1}^{n} A_{ip_i}$ ，我们所求的式子与行列式十分类似，被称为积和式（$permanent$），记作 $perm(A)$ ，我们发现 $perm(A) \equiv det(A) (mod \ 2)$ 

进一步地，模 $2$ 意义下行列式是否为 $0$ ，实际上是判断矩阵在模 $2$ 意义下是否可逆，即检查是否有向量能被其他向量在模 $2$ 意义下线性表示出

此时，区间的条件发挥了性质。我们可以建出一张 $n + 1$ 个点的图，对于区间 $[l_i, r_i]$，将 $l_i$ 与 $r_i + 1$ 连边。如果在图上$u, v$ 之间可达，说明通过向量模二意义下的线性运算，可以产生一个其对应的的区间。
