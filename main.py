from functionTools import *
from fit import *
from show import *
# 深度图中或点云中选择ROI 区域
xmin,ymin,xmax,ymax=[2928,156,3340,492]
xmin,ymin,xmax,ymax=[2740,772,2889,1442]
#2740,772,2889,1442 右側面
#1948,474,2279,1110 左側面
xmin,ymin,xmax,ymax=[1948,474,2279,1110]

# inputType='depthPic'
# inputType='randomGenerate'
inputType='ply'
if inputType=='depthPic':
    xmin, ymin, xmax, ymax = [2740, 772, 2889, 1442]
    # 2740,772,2889,1442 右側面
    # 1948,474,2279,1110 左側面
    # xmin, ymin, xmax, ymax = [1948, 474, 2279, 1110]
    x1, y1, z1=getTxtPoint('depthData.txt',xmin,ymin,xmax,ymax)
    x2,y2,z2=randomChoice(x1,y1,z1,400)

elif inputType=='ply':
    path='source/test.ply'
    print('ply files:',path)
    # x1,y1,z1=getPlyPoint(path)
    # plyPlot(x1, y1, z1)
    ##ply points
    xmin, ymin, xmax, ymax = [-0.5, -0.3, -0.178, 0.1]
    # xmin, ymin, xmax, ymax = [-0.178, -0.3, 0.75, 0.1]

    x1,y1,z1=getPlyPoint(path,xmin,ymin,xmax,ymax)
    x2,y2,z2=randomChoice(x1,y1,z1,400)
    #plyPlot(x2, y2, z2)
elif inputType=='randomGenerate':
    x2, y2, z2 = makePoint(xmin,ymin,xmax,ymax,a=1,b=0,d=0.0,pointsNum=400)
print('point nums:',len(z2))

x2,y2,z2=normalize(x2,y2,z2)

pointsNum=len(z2)
xmin,ymin,xmax,ymax=min(x2),min(y2),max(x2),max(y2)

# adata,bdata,cdata,ddata=svd(x2,y2,z2)
adata,bdata,cdata,ddata=paddleNN(x2,y2,z2,1)

# adata,bdata,cdata,ddata,nVector=pca3D(x2,y2,z2,svdSolver='full')
# print(quiver2direction(nVector))
# adata,bdata,cdata,ddata=-0.441147,0.54115123,1,0.4567488
# adata,bdata,cdata,ddata=-0.12170778959989548 0.731898844242096 1 0.09105458110570908
##l  0.797809898853302 0.06747536361217499 1 0.045177217572927475
print('adata,bdata,cdata,ddata,',adata,bdata,cdata,ddata)

matplot(adata,bdata,ddata,x2,y2,z2,testNum=10)


########################
# 引入numpy模块并创建两个三维向量x和y
print(quiver2direction([adata,bdata,1]))
#(21.20529360458012, -26.340669000214305, 34.92197536082232)
#(5.6091964913982935, -35.99981527025057, 36.57354005219686)
##l2 (-38.51978576618727, -3.019308641500217, 38.68279617874105)
##l1 (33.103264906331404, 0.4657960744497558, 33.10740320510461)