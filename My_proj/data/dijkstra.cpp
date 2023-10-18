#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 510;
int dist[N];    //dist[i]表示结点i到起点的距离
int g[N][N];    //g[i][j]表示结点i到结点j的边的长度
bool st[N];     //st[i]表示该结点是否确定了最小距离，1是确定，0是未确定
int n, m;


void dijkstra()
{
    memset(dist, 0x3f, sizeof dist);//把距离初始化为正无穷
    dist[1] = 0;
    
    int iter = n;
    while(iter--)//n个点，循环n次
    {
        int t = -1;
        //t随便初始化了一个不存在的结点，它最终用来存储未确定最小距离的结点，且该结点与其它结点相比目前到起点的距离最小
        for(int i = 1; i <= n; i++)
            if(st[i] == 0 && (t == -1 || dist[t] > dist[i]))
                t = i;
                
        st[t] = true;
        //用结点t依次取更新其它结点到起点的距离，dist[i] = min(dist[i], dist[t] + g[t][i]);
        for(int i = 1; i <= n; i++)
            if(st[i] == 0)
                dist[i] = min(dist[i], dist[t] + g[t][i]);
                
    }
    
}

int main()
{
    
    cin >> n >> m;
    
    memset(g, 0x3f, sizeof g);//将边先初始化为正无穷
    
    while(m--)
    {
        int x, y, z;
        cin >> x >> y >> z;
        g[x][y] = min(g[x][y], z);//存在重边
    }
    
    
    dijkstra();
    
    
    if(dist[n] == 0x3f3f3f3f)
        cout << "-1" << endl;
    else
        cout << dist[n] << endl;
    

    return 0;
}