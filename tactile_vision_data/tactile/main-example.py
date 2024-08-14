import PyTac3D
import time

SN = ''  # 传感器SN
frameIndex = -1 # 帧序号
sendTimestamp = 0.0 # 时间戳
recvTimestamp = 0.0 # 时间戳

# 用于存储三维形貌、三维变形场、三维分布力、三维合力、三维合力矩数据的矩阵
P, D, F, Fr, Mr = None, None, None, None, None

def Tac3DRecvCallback(frame, param):
    global SN, frameIndex, sendTimestamp, recvTimestamp, P, D, F
    
    print() # 换行
    
    print(param) # 显示自定义参数

    # 获取SN
    SN = frame['SN']
    print(SN)
    
    # 获取帧序号
    frameIndex = frame['index']
    print(frameIndex)
    
    # 获取时间戳
    sendTimestamp = frame['sendTimestamp']
    recvTimestamp = frame['recvTimestamp']

    # 获取标志点三维形貌
    # P矩阵（numpy.array）为400行3列，400行分别对应400个标志点，3列分别为各标志点的x，y和z坐标
    P = frame.get('3D_Positions')

    # 获取标志点三维位移场
    # D矩阵为（numpy.array）400行3列，400行分别对应400个标志点，3列分别为各标志点的x，y和z位移
    D = frame.get('3D_Displacements')

    # 获取三维分布力
    # F矩阵为（numpy.array）400行3列，400行分别对应400个标志点，3列分别为各标志点附近区域所受的x，y和z方向力
    F = frame.get('3D_Forces')

    # 获得三维合力
    # Fr矩阵为1x3矩阵，3列分别为x，y和z方向合力
    Fr = frame.get('3D_ResultantForce')

    # 获得三维合力矩
    # Mr矩阵为1x3矩阵，3列分别为x，y和z方向合力矩
    Mr = frame.get('3D_ResultantMoment')

# 创建Sensor实例，设置回调函数为上面写好的Tac3DRecvCallback，设置UDP接收端口为9988，数据帧缓存队列最大长度为5
tac3d = PyTac3D.Sensor(recvCallback=Tac3DRecvCallback, port=9988, maxQSize=5, callbackParam = 'test param')

# 等待Tac3D-Desktop端启动传感器并建立连接
tac3d.waitForFrame()

time.sleep(5) # 5s

# 发送一次校准信号（应确保校准时传感器未与任何物体接触！否则会输出错误的数据！）
tac3d.calibrate(SN)

time.sleep(5) #5s

# 获取frame的另一种方式：通过getFrame获取缓存队列中的frame
frame = tac3d.getFrame()
if not frame is None:
    print(frame['SN'])

time.sleep(5) #5s

# # 发送一次关闭传感器的信号（不建议使用）
# tac3d.quitSensor(SN)
    




    
