# **String**

****

## $Border$ 与 $Period$

$p$ 是 $S$ 的 $Period$ $\Leftrightarrow$ $|S| - p$ 是 $S$ 的 $Border$

### $Border$ 性质

$Border$不具有二分性 

$Border$传递性：$S$ 的 $Border$ 的 $Border$ 也是 $S$ 的 $Border$

一个串的 Border 数量是 $O(N)$ 个，但它们组成了 $O(logN)$ 个等差数列

### $Period$ 性质

周期定理：若 $p, q$ 均为串 $S$ 的周期，则 $(p, q)$ 也为 $S$ 的周期

### 矩阵的最小二维循环周期

对于一个 $n \times m$ 的矩阵，对每一行进行 $kmp$ 求出循环周期，取出当前长度为 $border$ 的子串，即 $p = next[p], period = m - p, map[period] ++$，找到出现次数为 $n$ 的且最短的长度 $a$ ；对于每一列，则找到出现次数为 $m$ 的且最短的长度 $b$ ，原矩阵的 $a \times b$ 的子矩阵即为最小二维循环周期

****

## $Border$ 树

### 性质

每个前缀 $Prefix[i]$ 的所有 Border：节点 $i$ 到根的链

哪些前缀有长度为 $x$ 的 $Border$：$x$ 的子树

求两个前缀的公共 $Border$ 等价于求 $LCA$    

****

## $Palindrome$

### 回文串与 $Border$

对于回文串 $S$ ，其回文前（后）缀等价于 $Border$ 

### 本质不同的回文串

对于一个字符串中本质不同的回文串的数量，每个位置上的字符最多对答案产生 $1$ 的贡献，即本质不同的回文串最多 $n$ 个

出现本质不同的字符串的位置在 $manacher$ 暴力更新区间的位置

****

## $Palindrome \  Automaton$

节点至多 $N$ 个，每个节点代表了一种回文串，$len$ 表示该回文串的长度，$cnt$ 表示以当前状态的最后一个字符为结尾的回文子串的数量

后继边：每个后继边上有一个字母，若 $u$ 的一个后继节点为 $v$ ，则 $len[v] = len[u] + 2$ 

失配边：每个节点都有一个失配边，若 $u$ 的一个失配节点为 $v$ ，则 $v$ 节点所代表的回文串是 $u$ 节点的最大 $Border$ ，即最长回文后缀

沿着某个节点一直跳 $link$，可以找到以当前节点为后缀的所有回文串

$suff$ 为最后一个插入的字符所在节点



每个本质不同回文串的最长回文后缀，其对应着最小的周期

如果一个长度为 $n$ 的回文串 $S$ 有一个长度为 $k$ 的回文后缀，那么它就有一个长度为 $n - k$ 的周期

如果一个字符串 $S$ 有长度为 $d_1$ 的周期和长度为 $d_2$ 的周期，并且 $d_1 + d_2 \leq n$ ，那么它有一个长度为 $gcd(d_1, d_2)$ 的周期

如果一个回文串可以被分成两段回文的后缀，那么它肯定是某个回文串 $T$ 重复大于等于 $2$ 次形成的，并且前后缀的长度都是 $|T|$ 的倍数

如果一个回文串有一个可以整除其长度 $n$ 的周期 $d$ ，且 $d \neq n$ ，那么它的最小周期也能整除 $n$ 

### 求回文子串出现次数

```c++
std::vector<int> p(n + 1);
p[0] = 1;
for (int i = 0; i < n; i++) {
    pam.add(s[i]);
    p[i + 1] = pam.suff;
}
const int N = pam.size();
std::vector<std::vector<int>> adj(N);
std::vector<int> freq(N);
for (int i = 0; i < n; i++) {
    freq[p[i + 1]]++;
}
for (int i = 2; i < N; i++) {
    adj[pam.link(i)].push_back(i);
}

auto dfs = [&](auto &&self, int u) -> void {
    for (auto v : adj[u]) {
        self(self, v);
        freq[u] += freq[v];
    }
};
dfs(dfs, 1);
i64 ans = 0;
for (int i = 2; i < N; i++) {
    ans += freq[i];
}
```

****

## $Suffix \  Array$

### 本质不同的子串

$\sum_i n - sa[i] + 1 - lc[i]$ 

按字典序从小到大枚举所有后缀，统计有多少个新出现的前缀

### 求长度大于 $K$ 的公共子串的数量

分别枚举 $S$ 串和 $T$ 串中的后缀，计算其包含的公共子串的数量，利用单调栈维护

### 最长公共子串

```c++
std::string s, t;
len = s.length();
for (int i = 0; i < n - 1; i++) {
    if (1LL * (sa[i] - len) * (sa[i + 1] - len) < 0) {
        ans = std::max(ans, lc[i]);
    }
}
```

### 本质不同的公共子串的数量

```c++
i64 ans = 0;
int last = 0;
for (int i = 0; i < n - 1; i++) {
    if (1ll * (sa[i] - len) * (sa[i + 1] - len) < 0) {
        ans += std::max(lc[i] - last, 0);
        last = lc[i];
    } else {
        last = std::min(last, lc[i]);
    }
}
```

### 判断一个串是否为其子串

二分查找

### 求所有子串的 $border$ 之和 /  $lcp$ 之和

```c++
std::vector<std::vector<int>> pos(n + 1);

i64 ans = 0;
for (int i = 0; i < n - 1; i++) {
    pos[lc[i]].emplace_back(i);
}   
i64 now = 0;

auto calc = [](int x) {
    return 1LL * x * (x + 1) / 2;
};

// border之和为lcp之和
// now 为 lcp[x][y] = i 的数对有多少个

for (int i = n; i; i--) {
    now++;
    for (auto v : pos[i]) {
        int pa = find(v), pb = find(v + 1);
        now -= calc(sz[pa]) + calc(sz[pb]);
        if (pa != pb) {
            p[pa] = pb;
            sz[pb] += sz[pa];
        }
        now += calc(sz[pb]);
    }
    ans += now;
}
```

### 求所有重复子串的重复次数

单调栈 / 单调队列

```c++
int len = n, pos = 0;
i64 max = n;

std::vector<int> l(n, -1), r(n, n);
std::vector<int> stk;    
for (int i = 0; i < n; i++) {
    while (!stk.empty() && lc[stk.back()] >= lc[i]) {
        stk.pop_back();
    }
    if (!stk.empty()) {
        l[i] = stk.back();
    }
    stk.emplace_back(i);
}

stk.clear();
for (int i = n - 1; i >= 0; i--) {
    while (!stk.empty() && lc[stk.back()] >= lc[i]) {
        stk.pop_back();
    }
    if (!stk.empty()) {
        r[i] = stk.back();
    }
    stk.emplace_back(i);
}

for (int i = 0; i < n; i++) {
    int cnt = r[i] - l[i];
    i64 res = 1LL * cnt * lc[i];
    if (res > max) {
        max = res;
        pos = sa[i];
        len = lc[i];
    }
}
```

### 第 $K$ 大子串

```c++
std::vector<int> len(n);
for (int i = 0; i < n; i++) {
    len[i] = n - SA.sa[i];
}

std::vector<std::vector<std::array<int, 3>>> seg(n);

if (type == 0) {
    for (int i = 0; i < n; i++) {
        seg[i].push_back({i == 0 ? 0 : SA.lc[i - 1], len[i], 1});
    }
} else {
    std::vector<std::vector<int>> pos(n);
    std::vector<int> f(n), siz(n, 1);
    std::iota(f.begin(), f.end(), 0);
    auto leader = [&](int x) {
        while (f[x] != x) {
            x = f[x] = f[f[x]];
        }
        return x;
    };

    for (int i = 0; i < n - 1; i++) {
        pos[SA.lc[i]].push_back(i);
    }
    for (int i = n - 1; i >= 0; i--) {
        for (auto p : pos[i]) {
            auto x = leader(p);
            auto y = leader(p + 1);

            if (len[x] > i) {
                seg[x].push_back({i, len[x], siz[x]});
            }
            if (len[y] > i) {
                seg[y].push_back({i, len[y], siz[y]});
            }

            siz[x] += siz[y];
            len[x] = i;
            f[y] = x;
        }
    }
    if (len[0] > 0) {
        seg[0].push_back({0, len[0], siz[0]});
    }

    for (auto &v : seg) {
        std::reverse(v.begin(), v.end());
    }

}

auto print = [&](int l, int r) {
    for (int i = l; i < r; i++) {
        std::cout << s[i];
    }
    std::cout << "\n";
};
for (int i = 0; i < n; i++) {
    for (auto [l, r, cnt] : seg[i]) {
        if (1LL * (r - l) * cnt < k) {
            k -= 1LL * (r - l) * cnt;
            continue;
        }
        int delta = (k - 1) / cnt;
        print(SA.sa[i], SA.sa[i] + l + 1 + delta);
        return 0;
    }
}
```



### 可重叠最长重复子串 / 不可重叠最长重复子串 / 可重叠 $k$ 次重复子串

### 回文串 / 平方串

****

## $Suffix \ Automaton$

每个状态对应一个 $Right$ 集合$(r_1,r_2,\dots,r_k)$，表示重复出现的相同的子串的右端点分别出现在 $r_i$ 的位置

所有节点的 $Right$ 集合互不相同

每个节点代表的串之间形成后缀关系，即所有串都是最长串的后缀

每个节点代表的串的长度时连续区间，记为 $[MinL(s), MaxL(s)]$ 

求 $|Right(s)|$  即子树求和

每个前缀所在的状态两两不同

共有 $|S|$ 个叶子节点，分别对应于每个前缀 $S[1,i]$ 所在的状态，即 $p[i]$ 

$link树$ 至多有 $2|S| - 1$ 个节点，即至多有这么多不同的 $Right$ 集合，即后缀自动机节点个数为 $O(2n)$ 

任意串 $w$ 的后缀全部位于 $s(w)$ 的后缀连接路径上

若某个状态 $s$ 拥有 $ch$ 的转移边，那么 $link(s)$ 也一定有 $ch$ 的转移边（但不一定转移到同一个状态）

每个状态 $s$ 的 $Right(s)$ 等价于他在后缀连接树子树的叶子节点集合

对于任意状态，$MaxL[link(s)] = MinL[s] - 1$ ，因此每个状态只需要额外记录 $len(s) = MaxL[s]$ 以及 $link(s)$ 

所有终止状态都能代表至少一个后缀

### 定位子串

每次询问给出 $l, r, a, b$ ，询问子串 $s[l ... r]$ 在子串 $s[a ... b]$  中出现次数

```c++
int query(Node *s, int l, int r, int x, int y) {
    if (x >= r || y <= l) {
        return 0;
    }
    if (l >= x && r <= y) {
        return s->sz;
    }
    int m = (l + r) / 2;
    return query(s->l, l, m, x, y) + query(s->r, m, r, x, y);
}
int main() {
    std::cin.tie(nullptr);
    std::ios_base::sync_with_stdio(false);

    std::string str;
    std::cin >> str;
    int n = str.length();

    std::vector<int> p(n + 1);
    p[0] = 1;
    SAM sam;
    for (int i = 0; i < n; i++) {
        p[i + 1] = sam.extend(p[i], str[i]);
    }
    const int N = sam.size();
    std::vector<std::vector<int>> adj(N);
    std::vector<int> freq(N);
    for (int i = 0; i < n; i++) {
        freq[p[i + 1]]++;
    }
    for (int i = 2; i < N; i++) {
        adj[sam.link(i)].emplace_back(i);
    }

    std::vector<int> dep(N);
    const int lg = std::__lg(N);
    std::vector f(lg + 1, std::vector(N, 1));
    std::vector<int> in(N), out(N), seq(N);
    int cur = 0;
    auto dfs = [&](auto &&self, int u) -> void {
        for (int i = 0; (2 << i) <= dep[u]; i++) {
            f[i + 1][u] = f[i][f[i][u]];
        }
        in[u] = cur++;
        seq[in[u]] = u;
        for (auto v : adj[u]) {
            if (v == f[0][u]) {
                continue;
            }
            f[0][v] = u;
            dep[v] = dep[u] + 1;
            self(self, v);
        }
        out[u] = cur;
    };
    dfs(dfs, 1);

    auto get = [&](int u, int len) {
        for (int i = lg; i >= 0; i--) {
            if (sam.len(f[i][u]) >= len) {
                u = f[i][u];
            }
        }
        return u;
    };

    std::vector<Node *> rt(cur + 1);
    std::vector<int> pos(N, -1);
    for (int i = 0; i < n; i++) {
        pos[p[i + 1]] = i;
    }
    rt[0] = build(0, n);
    for (int i = 0; i < cur; i++) {
        int j = seq[i];
        if (pos[j] == -1) {
            rt[i + 1] = rt[i];
        } else {
            rt[i + 1] = insert(rt[i], 0, n, pos[j]);
        }
    }

    int q;
    std::cin >> q;
    while (q--) {
        int l, r, a, b;
        std::cin >> l >> r >> a >> b;

        l--, a--;

        auto u = get(p[r], r - l);

        int x = in[u], y = out[u];
        if (r - l > b - a) {
            std::cout << 0 << "\n";
            continue;
        }
        std::cout << query(rt[y], 0, n, a + r - l - 1, b) - query(rt[x], 0, n, a + r - l - 1, b) << "\n";
    }

    return 0;
}
```



****
