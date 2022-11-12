import numpy as np
np.random.seed(17)
def makePoint(xmin,ymin,xmax,ymax,a=3,b=2,d=1,pointsNum=40):
    # 随机生成平面一上的点
    x1 = [np.random.randint(xmin, xmax) for i in range(pointsNum)]
    y1 = [np.random.randint(ymin, ymax) for i in range(pointsNum)]
    z1 = [xx * a + yy * b + d for xx, yy in zip(x1, y1)]
    return x1,y1,z1

def getTxtPoint(path,xmin,ymin,xmax,ymax):
    z2=np.loadtxt(path)
    # z2-=np.min(z2)
    height,width=z2.shape
    x2=np.ones(z2.shape)
    y2=np.ones(z2.shape)
    for index in range(width):
        x2[:,index]*=index

    for index in range(height):
        y2[index,:]*=index

    x2= list(x2[ymin:ymax,xmin:xmax].flatten().astype('int16'))
    y2= list(y2[ymin:ymax,xmin:xmax].flatten().astype('int16'))
    z2= list(z2[ymin:ymax,xmin:xmax].flatten().astype('int16'))
    return x2, y2,  z2
def randomChoice(x1,y1,z1,pointsNum=400):
    if len(z1)<pointsNum or pointsNum<=0:
        return x1, y1, z1

    else:
        indexList=np.random.randint(0,len(z1),[pointsNum,])
        x2=[x1[ii] for ii in indexList]
        y2 = [y1[ii] for ii in indexList]
        z2=[z1[ii] for ii in indexList]
        return x2,y2,z2


def getPlyPoint(fileName='45-45.ply',xmin=None,ymin=None,xmax=None,ymax=None):
    xList,yList,zList=[],[],[]
    from plyfile import PlyData
    plydata = PlyData.read(fileName)
    xList=np.array(plydata['vertex']['x'])
    yList=np.array(plydata['vertex']['y'])
    zList=np.array(plydata['vertex']['z'])
    if xmin is None or ymin is None or xmax is None or ymax is None:
        return xList, yList, zList
    else:
        ##
        indexList=np.where((xList>xmin)&(xList<xmax)&(yList>ymin)&(yList<ymax))
        x2 = [xList[ii] for ii in indexList][0]
        y2 = [yList[ii] for ii in indexList][0]
        z2 = [zList[ii] for ii in indexList][0]
        return x2, y2, z2
    # vetexColorList=plydata.elements[0].data
    # ''' vetexColorList=
    #  [(-0.62158203,  0.37597656, -0.98876953, 211, 3, 211)
    # (-0.6176758 ,  0.37475586, -0.98583984, 212, 3, 215)...'''
    # vetexColorArray=np.array(vetexColorList)
    # for vc in vetexColorList:
    #     xList.append(vc[0])
    #     yList.append(vc[1])
    #     zList.append(vc[1])
    # return xList,yList,zList

def normalize(x2,y2,z2,centerPoint=0):
    x2 = (np.array(x2 )- min(x2)) / (max(x2) - min(x2))
    y2 = (np.array(y2) - min(y2)) / (max(y2) - min(y2))
    z2 = (np.array(z2) - min(z2)) / (max(z2) - min(z2))
    if centerPoint==0:
        x2-=0.5
        y2-=0.5
        z2-=0.5
    return x2,y2,z2


def quiver2angle(q):
    Lq=np.sqrt(q[0]**2+q[1]**2+q[2]**2)
    angle_xa = np.arccos(q[0]/Lq)
    angle_ya = np.arccos(q[1]/Lq)
    angle_za = np.arccos(q[2]/Lq)
    angle_x= angle_xa * 180 / np.pi
    angle_y= angle_ya * 180 / np.pi
    angle_z = angle_za * 180 / np.pi
    ##x，y轴转向面向我们为0°，要转90°

    angle_x=angle_x-90

    angle_y=angle_y-90
    return angle_x,angle_y,angle_z

if __name__=='__main__':
    xmin,ymin,xmax,ymax=[2740,772,2889,1442]
    # x1, y1, z1 = getTxtPoint('depthData.txt',xmin,ymin,xmax,ymax)
    getPlyPoint('45-45.ply')