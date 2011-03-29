import OpenGL.GLUT as GLUT
import OpenGL.GL as GL
import OpenGL.GLU as GLU

import numpy as np
import sys

class gl_3d_window(object):
    def __init__(self, width=640, height=480,title="",refresh=30,render_func=lambda : None, rf_args=(), rf_kwargs = {}):
        #mouse handling for transforming scene
        self.mouse_down = False
        self.mouse_old = np.array([0., 0.])
        self.rotate = np.array([0., 0., 0.])
        self.translate = np.array([0., 0., 0.])
        self.scale = 1.0

        self.width = width
        self.height = height

        GLUT.glutInit(sys.argv)
        GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DOUBLE | GLUT.GLUT_DEPTH)
        GLUT.glutInitWindowSize(self.width, self.height)
        GLUT.glutInitWindowPosition(0, 0)
        self.win = GLUT.glutCreateWindow(title)

        #gets called by GLUT every frame
        GLUT.glutDisplayFunc(self.draw(render_func,*rf_args,**rf_kwargs))

        #handle user input
        GLUT.glutKeyboardFunc(self.on_key)
        GLUT.glutMouseFunc(self.on_click)
        GLUT.glutMotionFunc(self.on_mouse_motion)
        GLUT.glutMouseWheelFunc(self.on_mouse_wheel)
        
        #this will call draw every refresh ms
        GLUT.glutTimerFunc(refresh, self.timer, refresh)

        #setup OpenGL scene
        self.glinit()

        #set up initial conditions
        #create our OpenCL instance

        GLUT.glutMainLoop()
        

    def glinit(self):
        GL.glViewport(0, 0, self.width, self.height)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GLU.gluPerspective(60., self.width / float(self.height), .1, 1000.)
        GL.glMatrixMode(GL.GL_MODELVIEW)


    ###GL CALLBACKS
    def timer(self, t):
        GLUT.glutTimerFunc(t, self.timer, t)
        GLUT.glutPostRedisplay()

    def on_key(self, *args):
        ESCAPE = '\033'
        if args[0] == ESCAPE or args[0] == 'q':
            sys.exit()
        elif args[0] == 'r':
            self.rotate = np.array([0., 0., 0.])
            self.translate = np.array([0., 0., 0.])
            self.scale = 1.0

    def on_click(self, button, state, x, y):
        if state == GLUT.GLUT_DOWN:
            self.mouse_down = True
            self.button = button
            if self.button == 3:
                self.scale *= 1.1
            if self.button == 4:
                self.scale *= 0.9
        else:
            self.mouse_down = False
        self.mouse_old[0] = x
        self.mouse_old[1] = y

    
    def on_mouse_motion(self, x, y):
        dx = x - self.mouse_old[0]
        dy = y - self.mouse_old[1]
        if self.mouse_down and self.button == 0: #left button
            self.rotate[0] += dy * .2
            self.rotate[1] += dx * .2
        elif self.mouse_down and self.button == 2: #right button
            self.translate[0] += dx * .01
            self.translate[1] -= dy * .01
        self.mouse_old[0] = x
        self.mouse_old[1] = y

    def on_mouse_wheel(self, button, dir, x, y):
        print button
        print dir
        print x
        print y
        if dir > 0:
            self.scale *= .01
        else:
            self.scale *= 1.01


    ###END GL CALLBACKS


    def draw(self,f,*args,**kwargs):
        """Render the particles"""        
        #update or particle positions by calling the OpenCL kernel
        def draw_func():
            GL.glFlush()

            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glLoadIdentity()

            #handle mouse transformations
            GL.glTranslatef(0.,0.,-2)
            GL.glRotatef(self.rotate[0], 1, 0, 0)
            GL.glRotatef(self.rotate[1], 0, 1, 0) #we switched around the axis so make this rotate_z
            GL.glTranslatef(self.translate[0], self.translate[1], self.translate[2])
            GL.glScale(self.scale,self.scale,self.scale)
        
            #render the particles
            f(*args,**kwargs)

            GLUT.glutSwapBuffers()
        return draw_func

def rend_func(pos,col,p_size):
    import OpenGL.GL as GL
    GL.glEnable(GL.GL_POINT_SMOOTH)
    GL.glPointSize(p_size)
    GL.glEnable(GL.GL_BLEND)
    GL.glBlendFunc(GL.GL_SRC_ALPHA,GL.GL_ONE_MINUS_SRC_ALPHA)
    #col = np.random.random_sample((num,4))
    #pos = np.random.random_sample((num,4))
    GL.glColorPointer(4,GL.GL_FLOAT,0,col)
    GL.glVertexPointer(4,GL.GL_FLOAT,0,pos)
    GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
    GL.glEnableClientState(GL.GL_COLOR_ARRAY)
    GL.glDrawArrays(GL.GL_POINTS,0,len(pos))
    GL.glDisableClientState(GL.GL_COLOR_ARRAY)
    GL.glDisableClientState(GL.GL_VERTEX_ARRAY)

if __name__ == "__main__":
    num = int(sys.argv[1])
    refresh = int(sys.argv[2])
    try:
        p_size = int(sys.argv[3])
    except:
        p_size = 5
    pos = np.ndarray((num,4),dtype=np.float32)
    pos[:,0] = np.random.random_sample((num,)) - 0.5
    pos[:,1] = np.random.random_sample((num,)) - 0.5
    pos[:,2] = np.random.random_sample((num,)) - 0.5
    pos[:,3] = 0.75
    col = np.ndarray((num,4),dtype=np.float32)
    rand_col = np.sqrt(pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)/np.sqrt(3*(.5**2))
    col[:,0] = np.min([4*rand_col - 1.5,-4 * rand_col + 4.5],axis=0)
    col[:,1] = np.min([4*rand_col - 0.5,-4 * rand_col + 3.5],axis=0)
    col[:,2] = np.min([4*rand_col + 0.5,-4 * rand_col + 2.5],axis=0)
    col[:.3] = 1.0
    
    gl_3d_window(render_func = rend_func,rf_args=(pos,col,p_size),refresh=refresh)
