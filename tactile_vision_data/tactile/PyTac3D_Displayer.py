import PyTac3D
import ruamel.yaml
import vedo
import numpy as np

class Tac3D_Displayer:
    def __init__(self, port=9988):
        self._scaleF = 100
        self._scaleD = 10
        self._GenConnect(20, 20)
        
        self._recvFirstFrame = False
        self.Tac3DSensor = PyTac3D.Sensor(port=port, maxQSize=1)
        self.SN = ''
        
        self._plotter = vedo.Plotter(N=2)
        
        self._box = vedo.Box(pos=(0,0,0), length=16, width=16, height=8).alpha(0.03)
        self._axs = vedo.Axes(self._box, c='k', xygrid=False)  # returns an Assembly object

        self._enable_Mesh0 = True
        self._enable_Displacements = True
        self._enable_Mesh1 = True
        self._enable_Forces = True

        self._refPoints_org = None

        self._button_calibrate = self._plotter.at(0).add_button(
                    self._ButtonFunc_Calibrate,
                    states=["calibrate"],
                    font="Kanopus",
                    pos=(0.3, 0.92),
                    size=32,
                )
        self._button_mesh0 = self._plotter.at(0).add_button(
                    self._ButtonFunc_Mesh0,
                    states=["\u23F8 Geometry","\u23F5 Geometry"],
                    font="Kanopus",
                    pos=(0.3, 0.05),
                    size=32,
                )
        self._button_displacements = self._plotter.at(0).add_button(
                    self._ButtonFunc_Displacements,
                    states=["\u23F8 Displacements","\u23F5 Displacements"],
                    font="Kanopus",
                    pos=(0.7, 0.05),
                    size=32,
                )
                
        self._button_mesh1 = self._plotter.at(1).add_button(
                    self._ButtonFunc_Mesh1,
                    states=["\u23F8 Geometry","\u23F5 Geometry"],
                    font="Kanopus",
                    pos=(0.3, 0.05),
                    size=32,
                )
        self._button_force = self._plotter.at(1).add_button(
                    self._ButtonFunc_Forces,
                    states=["\u23F8 Forces","\u23F5 Forces"],
                    font="Kanopus",
                    pos=(0.7, 0.05),
                    size=32,
                )


    def Run(self):
        self._plotter.at(0).show()
        self._plotter.at(1).show()
        self._timerevt = self._plotter.add_callback('timer', self._ShowFrame)
        self._timer_id = self._plotter.timer_callback('create', dt=10)
        self._plotter.interactive().close()
        
    def _ShowFrame(self, event):
        frame = self.Tac3DSensor.getFrame()
        if not frame is None:
            self.SN = frame['SN']
            
            L = frame.get('3D_Positions')
            D = frame.get('3D_Displacements')
            F = frame.get('3D_Forces')

            self._plotter.at(0).clear()
            self._plotter.at(0).add(self._box, self._axs)
            self._plotter.at(1).clear()
            self._plotter.at(1).add(self._box, self._axs)
            
            if not L is None:
                mesh = vedo.Mesh([L, self._connect], alpha=0.9, c=[150,150,230])
                if self._enable_Mesh0:
                    self._plotter.at(0).add(mesh)
                if self._enable_Displacements and not D is None:
                    arrsD = vedo.Arrows(list(L), list(L+D*self._scaleD), s=2)
                    self._plotter.at(0).add(arrsD)
                if self._enable_Mesh1:
                    self._plotter.at(1).add(mesh)
                if self._enable_Forces and not F is None:
                    arrsF = vedo.Arrows(list(L), list(L+F*self._scaleF), s=2)
                    self._plotter.at(1).add(arrsF)
            
            refPoint = frame.get('3D_refPoints')

            if not refPoint is None:
                if self._refPoints_org is None:
                    self._refPoints_org = refPoint
                    
                refP = vedo.Points(refPoint, c=[0,0,0])
                refP0 = vedo.Points(self._refPoints_org, c=[255,0,0])
                self._plotter.at(0).add(refP, refP0)
                
            
            self._plotter.at(0).render()
            self._plotter.at(1).render()
            
            if not self._recvFirstFrame:
                self._recvFirstFrame = True
                self._plotter.reset_camera()
    
    def _GenConnect(self, nx, ny):
        self._connect = []
        for iy in range(ny-1):
            for ix in range(nx-1):
                idx = iy * nx + ix
                self._connect.append([idx, idx+1, idx+nx])
                self._connect.append([idx+nx+1, idx+nx, idx+1])
        
    def _ButtonFunc_Calibrate(self):
        if not self.Tac3DSensor.frame is None:
            self.Tac3DSensor.calibrate(self.Tac3DSensor.frame.get('SN'))
        
    def _ButtonFunc_Mesh0(self):
        self._button_mesh0.switch()
        self._enable_Mesh0 = not self._enable_Mesh0
        
    def _ButtonFunc_Displacements(self):
        self._button_displacements.switch()
        self._enable_Displacements = not self._enable_Displacements
        
    def _ButtonFunc_Mesh1(self):
        self._button_mesh1.switch()
        self._enable_Mesh1 = not self._enable_Mesh1
        
    def _ButtonFunc_Forces(self):
        self._button_force.switch()
        self._enable_Forces = not self._enable_Forces
        
if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        port = int(sys.argv[1])
    else:
        port = 9988
    displayer = Tac3D_Displayer(port)
    displayer.Run()
