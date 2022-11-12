import numpy as np
import matplotlib.pyplot as plt

def plyPlot(x2,y2,z2):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.scatter(x2, y2, z2, c='r', marker='o')
    plt.show()

def matplot(adata,bdata,ddata,x2,y2,z2,testNum=100,title="points and plane"):
    x_train=(x2[:-testNum])
    y_train=(y2[:-testNum])
    z_train=(z2[:-testNum])
    x_test=(x2[-testNum:])
    y_test=(y2[-testNum:])
    z_test=(z2[-testNum:])
    fig = plt.figure()
    try:
        ax = fig.add_subplot(111, projection='3d')
    except:
        from mpl_toolkits.mplot3d import axes3d, Axes3D
        ax = Axes3D(fig)
    #
    plt.title('《plane vs source points》fit：'+title, fontproperties="SimHei", fontsize=10)
    ax.scatter(x_train, y_train, z_train)
    # ax.scatter(x_test, y_test, z_test,c='#DC143C')
    #adata=-0.19717
    #bdata=-0.40139
    #ddata=0.826673
    # print(adata,bdata,ddata)
    x = np.arange(np.min(x2), np.max(x2), (np.max(x2)-np.min(x2))/100)
    y = np.arange(np.min(y2), np.max(y2), (np.max(y2)-np.min(y2))/100)

    # 生成网格数据
    X, Y = np.meshgrid(x, y)
    Z = adata * X + bdata * Y + ddata
    ## show in difference way
    surfaceFlag=False
    if surfaceFlag:
        ax.plot_surface(X,Y,Z,
                        color='r',
                        alpha=0.6
                        )
    else:
        ax.plot_wireframe(X,Y,Z, rstride=10, cstride=10,color='g')
    '''
    rowLength=1
    ox=oy=0.5
    oz=(ox*adata+oy*bdata+ddata)
    print('ox,oy, oz, adata+ox, bdata+oy, 1+oz',ox,oy, oz, adata+ox, bdata+oy, 1+oz)
    ax.quiver(ox,oy, oz, adata+ox, bdata+oy, 1+oz, length=rowLength, normalize=True)

    ox=0.5
    oy=0.3
    oz=(ox*adata+oy*bdata+ddata)
    ax.quiver(ox,oy, oz, adata+ox, bdata+oy, 1+oz, length=rowLength, normalize=True)

    ox=0.3
    oy=0.5
    oz=(ox*adata+oy*bdata+ddata)
    ax.quiver(ox,oy, oz, adata+ox, bdata+oy, 1+oz, length=rowLength, normalize=True)
    ox=0.3
    oy=0.3
    oz=(ox*adata+oy*bdata+ddata)
    ax.quiver(ox,oy, oz, adata+ox, bdata+oy, 1+oz, length=rowLength, normalize=True)
    '''
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
